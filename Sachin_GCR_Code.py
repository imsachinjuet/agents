import os
import re
import uuid
from typing import Optional, Dict, Any, List
import logging
import boto3
from botocore.exceptions import ClientError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi import FastAPI, Response
from pydantic import BaseModel

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from pinecone import Pinecone, ServerlessSpec, PineconeApiException
from langchain_core.prompts import ChatPromptTemplate
import openai
import pandas as pd
from dotenv import dotenv_values

from dotenv import load_dotenv
load_dotenv()

import os
import logging



LOG_FILENAME = "session_debug.log"

# Delete previous log file at startup (if exists)
if os.path.exists(LOG_FILENAME):
    try:
        os.remove(LOG_FILENAME)
    except Exception as e:
        print(f"Could not delete old log file: {e}")



# Set up logging to file (all logs go here)
file_handler = logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # Log everything to file
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(file_formatter)

# Optionally, set up a minimal stream handler (terminal)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)  # Only show WARNING and above in terminal
stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
stream_handler.setFormatter(stream_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Minimum level for root logger
    handlers=[file_handler, stream_handler],
)

logger = logging.getLogger(__name__)


# # right up at the top of your file, define:
# FORBIDDEN_OFFICES = [
#     "MFHD Program Office", "SFHD Program Office", "ORCF Program Office",
#     "OHF Program Office", "Coalfire", "OCU Program Office",
#     "Sample Program office", "OPPAD", "QA Test", "Testofficemanagement",
#     "Test Office Manag", "Test Office Management", "Test Office Management12",
#     "Test Office Management34", "Test Office Management67",
#     "test900", "QA9899", "Mobile test"
# ]

# def get_company_id_from_question(question: str, user_company_id: int) -> int | None:
#     import re

#     q_lower = question.lower()

#     # 0) If the user literally mentions *any* of those forbidden office names, block:
#     for name in FORBIDDEN_OFFICES:
#         if name.lower() in q_lower:
#             return -1

#     # 1) If they mentioned no 'program office' phrase and no 'company_id', assume their own:
#     if not re.search(r'\b(program office|company[_ ]?id)\b', question, re.IGNORECASE):
#         return user_company_id

#     # 2) Explicit company_id check
#     match = re.search(r'company[\s_]?id\s*=?\s*(\d+)', question, re.IGNORECASE)
#     if match:
#         found = int(match.group(1))
#         return found if found == user_company_id else -1

#     # 3) LLM‑based cross‑company classifier as fallback
#     try:
#         llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)
#         prompt = ChatPromptTemplate.from_template(LLM_CROSS_COMPANY_PROMPT)
#         result = (prompt | llm).invoke({
#             "question": question,
#             "company_id": user_company_id
#         }).content.strip().upper()
#         if result == "YES":
#             return -1
#     except Exception:
#         pass

#     # 4) Otherwise, it’s safe to proceed under their own company
#     return user_company_id


SUMMARY_KEYWORD_PROMPT = """
You are an AI assistant.  Determine whether the user is asking to search for a keyword inside existing document *summaries*.

**DO NOT consider any IREMS number or IREMS or anything related to IREMS inside this summary.**

- If **yes**, reply exactly with:
  `SEARCH_SUMMARY_KEYWORD:{{keyword}}`
  where `{{keyword}}` is the phrase they want to match (without any extra quotes).

- Otherwise reply with exactly:
  `NO`

User message: {question}
"""

def extract_summary_keyword_via_llm(question: str, openai_api_key: str) -> str | None:
    prompt = ChatPromptTemplate.from_template(SUMMARY_KEYWORD_PROMPT)
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=openai_api_key)
    resp = (prompt | llm).invoke({"question": question}).content.strip()
    if resp.startswith("SEARCH_SUMMARY_KEYWORD:"):
        return resp.split(":", 1)[1].strip()
    return None


def search_summaries(keyword: str, company_id: int, engine) -> List[str]:
    sql = text("""
        SELECT d.actual_file_name
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d
          ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE :kw
          AND s.file_plan_id = :cid
        ORDER BY d.upload_date DESC
        LIMIT 50
    """)
    # ✅ acquire a connection and execute there
    with engine.connect() as conn:
        result = conn.execute(sql, {
            "kw": f"%{keyword}%",
            "cid": company_id
        })
        rows = result.fetchall()
    return [r[0] for r in rows]

PERMISSION_CLASSIFICATION_PROMPT = """
You are an AI assistant.  Determine whether the user is asking about permissions
(read, write, edit, view, access rights, privileges, authorization).

- If **yes**, reply exactly with `YES`
- Otherwise reply exactly with `NO`

User message: {question}
"""




DOWNLOAD_CLASSIFICATION_PROMPT = """
You are an AI assistant. Given a user's message, determine if they are requesting to download a file (by filename, such as a PDF).
- If so, extract the exact filename (including the extension).
- If not, reply with ONLY "NO".

User message: {question}

Reply ONLY with:
- "DOWNLOAD:{{filename}}" (if the user is requesting a file download and you can extract the filename), or
- "NO" (if not a download request).
"""

def is_permission_question(question: str) -> bool:
    prompt = ChatPromptTemplate.from_template(PERMISSION_CLASSIFICATION_PROMPT)
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)
    resp = (prompt | llm).invoke({"question": question}).content.strip().upper()
    return resp == "YES"


def extract_download_filename_via_llm(question: str, openai_api_key: str) -> Optional[str]:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(DOWNLOAD_CLASSIFICATION_PROMPT)
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=openai_api_key)
    response = (prompt | llm).invoke({"question": question}).content.strip()
    if response.startswith("DOWNLOAD:"):
        filename = response.split(":", 1)[1].strip()
        if filename:
            return filename
    return None

def lookup_file_info(filename: str, company_id: int, engine):
    """
    Looks up actual_file_name and modified_file_path in mst_documents.
    Returns (actual_file_name, modified_file_path) or (None, None).
    """
    with engine.connect() as conn:
        query = text("""
            SELECT actual_file_name, modified_file_path FROM mst_documents
            WHERE actual_file_name = :filename AND company_id = :company_id
            LIMIT 1
        """)
        row = conn.execute(query, {"filename": filename, "company_id": company_id}).fetchone()
        if row:
            return row[0], row[1]
    return None, None

def generate_presigned_url(bucket_name, s3_path, region_name, expires_in=3600):
    """
    Generates a presigned S3 URL for the given file path.
    """
    s3_client = boto3.client('s3', region_name=region_name)
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket_name, 'Key': s3_path},
            ExpiresIn=expires_in
        )
        return url
    except ClientError as e:
        logger.exception("Failed to generate presigned URL")
        return None

def handle_download_request_llm(question: str, company_id: int, engine, openai_api_key: str) -> Optional[str]:
    filename = extract_download_filename_via_llm(question, openai_api_key)
    if filename:
        actual, s3_path = lookup_file_info(filename, company_id, engine)
        if not actual or not s3_path:
            return f"File '{filename}' not found for your company."
        # Compose full S3 path with prefix if not present
        env = dotenv_values(".env")
        bucket = env.get("bucket_name") or os.getenv("bucket_name") or os.getenv("BUCKET_NAME")
        region = env.get("region_name") or os.getenv("region_name") or os.getenv("REGION_NAME")
        prefix = env.get("environment_prefix") or os.getenv("environment_prefix") or os.getenv("ENVIRONMENT_PREFIX") or ""
        if prefix and not s3_path.startswith(prefix):
            s3_key = f"{prefix}/{s3_path}".replace("//", "/")
        else:
            s3_key = s3_path
        presigned_url = generate_presigned_url(bucket, s3_key, region)
        if not presigned_url:
            return "Sorry, unable to generate download link for this file."
        return f"Click to download: [{actual}]({presigned_url})"
    return None


import atexit

def remove_logfile():
    try:
        if os.path.exists(LOG_FILENAME):
            os.remove(LOG_FILENAME)
    except Exception as e:
        print(f"Could not remove log file at exit: {e}")

atexit.register(remove_logfile)

REQUIRED_ENV = [
    "OPENAI_API_KEY",
    "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DB",
    "PINECONE_API_KEY", "PINECONE_ENV", "PINECONE_INDEX", "PINECONE_CONVO_INDEX"
]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    raise EnvironmentError(f"Missing required environment vars: {', '.join(missing)}")

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
MYSQL_USER       = os.getenv("MYSQL_USER")
MYSQL_PASSWORD   = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST       = os.getenv("MYSQL_HOST", "localhost")
MYSQL_DB         = os.getenv("MYSQL_DB")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
PINECONE_CONVO_INDEX = os.getenv("PINECONE_CONVO_INDEX", "chatbot-conversation-global")

openai.api_key = OPENAI_API_KEY

sql_url = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    database=MYSQL_DB,
)
engine = create_engine(sql_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


SQL_CLASSIFICATION_PROMPT = """
You are an AI assistant that checks if a user input is a SQL command or attempts to run SQL statements (such as SELECT, INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, etc.).

If the input is a SQL command or an attempt to run SQL, respond with ONLY "YES".
If it is a natural language question, respond with ONLY "NO".

User input: {question}
Is this a SQL command? (YES/NO)
"""

SMALL_TALK_CLASSIFICATION_PROMPT = """
You are an AI assistant. If the user's message is purely small‑talk or a personal chit‑chat
(greetings like “hi”, “hello”, “hey”, or personal questions like “how are you?”, “who are you?”, 
“what’s your name?”, “tell me about yourself”), respond with ONLY “YES”. 
Otherwise, respond with ONLY “NO”.

User message: {question}
"""



def is_sql_command_via_llm(question: str) -> bool:
    logger.debug(f"Checking if question is a SQL command: {question}")
    llm = ChatOpenAI(
        model_name="gpt-4.1",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(SQL_CLASSIFICATION_PROMPT)
    chain = prompt | llm
    try:
        response = chain.invoke({"question": question}).content.strip().upper()
        logger.debug(f"LLM SQL classification response: {response}")
        return response == "YES"
    except Exception as e:
        logger.exception("Error running is_sql_command_via_llm")
        return False


def is_readonly_sql(query: str) -> bool:
    logger.debug(f"Checking if SQL query is read-only: {query}")
    if not query:
        return True
    q = re.sub(r"\s+", " ", query).strip().lower()
    if q.startswith(("select", "show", "describe", "explain")):
        forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "replace ", "truncate ", "rename ", "grant ", "revoke "]
        if ";" in q:
            first = q.split(";")[0]
            if any(f in first for f in forbidden):
                logger.warning(f"Query {query} contains forbidden statement in first part.")
                return False
        if any(f in q for f in forbidden):
            logger.warning(f"Query {query} contains forbidden statement.")
            return False
        return True
    return False

from sqlalchemy.engine.base import Connection
orig_conn_execute = Connection.execute
def readonly_conn_execute(self, statement, *args, **kwargs):
    if not is_readonly_sql(str(statement)):
        raise Exception("Only read-only SQL statements are allowed.")
    return orig_conn_execute(self, statement, *args, **kwargs)
Connection.execute = readonly_conn_execute

# --------- WRAPPER TO INJECT company_id FILTER -----------

def inject_company_id_filter(sql, company_id, engine):
    logger.debug(f"inject_company_id_filter called with sql={sql}, company_id={company_id}")
    inspector = inspect(engine)
    table_match = re.search(r'from\s+([`"]?)(\w+)\1', sql, re.IGNORECASE)
    if not table_match:
        logger.debug("No table match in SQL.")
        return sql
    table = table_match.group(2)
    cols = [c['name'] for c in inspector.get_columns(table)]
    if "company_id" not in cols:
        logger.debug(f"Table {table} has no company_id column.")
        return sql
    if re.search(r'company_id\s*=\s*', sql, re.IGNORECASE):
        logger.debug(f"company_id filter already present in sql: {sql}")
        return sql
    if re.search(r'\bwhere\b', sql, re.IGNORECASE):
        sql = re.sub(
            r'(\bwhere\b)',
            r'\1 company_id = {cid} AND '.format(cid=company_id),
            sql,
            count=1,
            flags=re.IGNORECASE
        )
    else:
        sql = re.sub(
            r'(from\s+[`\w"]+)',
            r'\1 WHERE company_id = {cid}'.format(cid=company_id),
            sql,
            count=1,
            flags=re.IGNORECASE
        )
    logger.debug(f"inject_company_id_filter output: {sql}")
    return sql

def make_company_id_sql_tool(orig_tool, company_id, engine):
    def tool_wrapper(input_str):
        logger.info(f"sql_db_query tool invoked with input: {input_str}")
        sql = input_str
        params = {}
        if isinstance(sql, dict):
            params = sql.get('params', {})
            sql = sql.get('query', sql)
        sql = inject_company_id_filter(sql, company_id, engine)

        # --- Intercept summary request ---
        if (
            "SELECT" in sql.upper() and
            "DOC_SUMMARY" in sql.upper() and
            "FROM MST_DOCUMENTS_SQS_UPLOAD" in sql.upper()
        ):
            import re
            if "file_plan_id" not in sql.lower():
                sql = re.sub(
                    r"SELECT\s+doc_summary",
                    "SELECT doc_summary, file_plan_id",
                    sql,
                    flags=re.IGNORECASE
                )
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(sql + " LIMIT 10"), params).fetchall()
                    rows = [dict(r._mapping) for r in result]
                found = False
                for row in rows:
                    if str(row.get("file_plan_id")) == str(company_id):
                        found = True
                        break
                if found:
                    logger.info("Returning doc_summary from mst_documents_sqs_upload.")
                    return {"doc_summary": row.get("doc_summary")}
                else:
                    logger.info("Summary not found for this company_id.")
                    return {"error": "Summary is not available in this program office."}
            except Exception as e:
                logger.exception("Error executing summary SQL in sql_db_query tool")
                return {"error": str(e)}

        # For all other SQL, run as normal
        try:
            return orig_tool(sql)
        except Exception as e:
            logger.exception("Error executing orig_tool in sql_db_query tool")
            return {"error": str(e)}

    return Tool(
        name=orig_tool.name,
        func=tool_wrapper,
        description=orig_tool.description
    )



# Only these tables will be searched for text columns
SEARCH_TABLES = [
   "mst_documents","mst_property","mst_documents_sqs_upload",
        "mst_users","mst_company","entity_transactions","mst_indexes",
        "mst_document_types","mst_app_urls","folder_mapping",
        "file_plan_template","mst_folders","entity_payments",
        "doc_index_mapping","mst_subscription_plan","mst_document_privileges",
        "role_app_urls_mapping","mst_roles","document_privileges"
]


ID_SEARCH_COLS = {"id", "index_value", "document_seq_id", "created_By"}

def search_anywhere(id_value: str, company_id: int) -> Dict[str, Any]:
    logger.info(f"search_anywhere called with id_value={id_value}, company_id={company_id}")
    inspector = inspect(engine)
    for table in SEARCH_TABLES:
        cols = [c["name"] for c in inspector.get_columns(table)]
        for col in cols:
            if col.lower().endswith("_id") or col in ID_SEARCH_COLS:
                sql = f"SELECT * FROM `{table}` WHERE `{col}` = :id"
                params = {"id": id_value}
                if "company_id" in cols:
                    sql += " AND company_id = :company_id"
                    params["company_id"] = company_id
                if not is_readonly_sql(sql):
                    logger.warning("search_anywhere detected non-readonly query.")
                    return {
                        "error": "Sorry, I am only allowed to read data. No changes or actions can be made in the database."
                    }
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(sql), params).fetchall()
                    if result:
                        logger.info(f"search_anywhere: found in table {table} column {col}")
                        return {
                            "table": table,
                            "column": col,
                            "rows": [dict(r._mapping) for r in result]
                        }
                except Exception as e:
                    logger.exception(f"search_anywhere: error running SQL for table={table}, col={col}")
    logger.info("search_anywhere: not found for any table/col.")
    return {
        "error": None,
        "chatbot_response": "this information is not available in the current program office"
    }


from concurrent.futures import ThreadPoolExecutor, as_completed

def search_table_string_columns(table, keyword, company_id, engine, limit):
    results = []
    from sqlalchemy import inspect
    inspector = inspect(engine)
    try:
        cols = inspector.get_columns(table)
        text_cols = [c["name"] for c in cols if c["type"].__class__.__name__.lower() in ("varchar", "text", "nvarchar", "char")]
        all_col_names = [c["name"] for c in cols]

        for col in text_cols:
            # For mst_documents, only search actual_file_name and return full row
            if table == "mst_documents":
                if col != "actual_file_name":
                    continue
                sql = f"SELECT * FROM `{table}` WHERE `{col}` LIKE :kw AND company_id = :cid LIMIT :limit"
            else:
                # For other tables, search all string columns and return all columns
                col_csv = ", ".join(f"`{colname}`" for colname in all_col_names)
                sql = f"SELECT {col_csv} FROM `{table}` WHERE `{col}` LIKE :kw AND company_id = :cid LIMIT :limit"

            params = {"kw": f"%{keyword}%", "cid": company_id, "limit": limit}
            with engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            if rows:
                results.append({
                    "table": table,
                    "column": col,
                    "rows": [dict(r._mapping) for r in rows]
                })
    except Exception as e:
        print(f"Error in table {table}: {e}")
    return results


def keyword_search(keyword: str, company_id: int, engine, limit=3):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for table in SEARCH_TABLES:
            futures.append(executor.submit(search_table_string_columns, table, keyword, company_id, engine, limit))
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.extend(res)
    return results



# def handle_keyword_results(results):
#     if not results:
#         return "No records found for your search."
#     if len(results) == 1 and len(results[0]['rows']) == 1:
#         row = results[0]['rows'][0]
#         return f"Found in table '{results[0]['table']}' (column '{results[0]['column']}'):\n" + "\n".join(f"{k}: {v}" for k, v in row.items())
#     summary = []
#     for res in results:
#         summary.append(f"{len(res['rows'])} results in '{res['table']}' (column '{res['column']}')")
#     response = "Multiple matches found:\n" + "\n".join(summary)
#     response += "\n\nPlease specify which table/type you are interested in for more details."
#     return response

db = SQLDatabase(
    engine,
    include_tables=[
        "mst_documents","mst_property","mst_documents_sqs_upload",
        "mst_users","mst_company","entity_transactions","mst_indexes",
        "document_types","mst_app_urls","folder_mapping",
        "file_plan_template","mst_folders","entity_payments",
        "doc_index_mapping","mst_subscription_plan","mst_document_privileges",
        "role_app_urls_mapping","mst_roles","document_privileges"
    ],
)
llm_sql = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4.1")
toolkit = SQLDatabaseToolkit(db=db, llm=llm_sql)
sql_tools = toolkit.get_tools()

def handle_pagination(
    engine,
    base_query: str,
    params: dict,
    order_by: str = "upload_date",
    order_dir: str = "DESC",
    page: int = 1,
    per_page: int = 10
    ) -> dict:
    logger.info(f"handle_pagination: base_query={base_query}, params={params}, page={page}, per_page={per_page}")
    per_page = 10
    if not re.search(r"order\s+by", base_query, re.IGNORECASE):
        ordered_query = f"{base_query} ORDER BY {order_by} {order_dir}"
    else:
        ordered_query = base_query
    count_sql = f"SELECT COUNT(*) AS cnt FROM ({base_query}) AS sub"
    if not is_readonly_sql(base_query):
        logger.error("handle_pagination: Non-readonly query detected.")
        return {"error": "Sorry, I am only allowed to read data. No changes or actions can be made in the database."}
    try:
        with engine.connect() as conn:
            total = conn.execute(text(count_sql), params).scalar()
        offset = (page - 1) * per_page
        page_sql = f"{ordered_query} LIMIT :limit OFFSET :offset"
        page_params = {**params, "limit": per_page, "offset": offset}
        df_page = pd.read_sql(text(page_sql), engine, params=page_params)
        logger.info(f"Pagination: total={total}, returning page {page}")
        return {
            "total": int(total),
            "page": page,
            "per_page": per_page,
            "rows": df_page.to_dict(orient="records"),
        }
    except Exception as e:
        logger.exception("handle_pagination: Exception during pagination")
        return {"error": str(e)}


# ----------------- Pinecone Setup --------------------
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX not in pc.list_indexes():
    try:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            pass
    except PineconeApiException as e:
        if e.status != 409:
            raise
index = pc.Index(PINECONE_INDEX)

PINECONE_LAST_MATCH: dict[str, dict] = {}
PINECONE_LAST_SCORE: dict[str, float] = {}

def get_actual_file_name_from_s3_key(s3_key: str, engine) -> str | None:
    query = text("""
        SELECT actual_file_name FROM mst_documents
        WHERE modified_file_path = :s3_key
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"s3_key": s3_key}).fetchone()
        if row and row[0]:
            return row[0]
    return None

def pinecone_semantic_search(
    question: str,
    top_k: int = 10,  # Increase to consider more potential matches
    namespace: str = "gcrdev",
    conv_id: str | None = None,
    min_score: float = 0.80
) -> str:
    logger.info(f"pinecone_semantic_search: question={question}, conv_id={conv_id}")
    try:
        # Step 1: Embed the question using OpenAI embeddings
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=[question]
        )
        emb = resp.data[0].embedding

        # Step 2: Query Pinecone index
        pinecone_response = index.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        matches = pinecone_response.get("matches", [])

        # Step 3: Filter relevant matches based on the minimum score
        relevant_matches = [match for match in matches if match["score"] >= min_score]

        if not relevant_matches:
            logger.info("pinecone_semantic_search: No relevant answer found.")
            return "[From The Document Content:] No relevant answer found in Pinecone index."

        # Optionally store last matches for the conversation
        if conv_id:
            PINECONE_LAST_SCORE[conv_id] = max(match["score"] for match in relevant_matches)
            PINECONE_LAST_MATCH[conv_id] = [match.get("metadata", {}) for match in relevant_matches]

        # Step 4: Format response with multiple documents
        response_parts = []
        for match in relevant_matches:
            meta = match.get('metadata', {})
            raw_text = meta.get('text', '[no text in metadata]')

            # Lookup actual file name using modified_file_path → actual_file_name
            s3_key = meta.get('s3_key')
            if s3_key:
                actual_file_name = get_actual_file_name_from_s3_key(s3_key, engine)
                if actual_file_name:
                    source_info = f"Actual File Name: **{actual_file_name}**"
                else:
                    # source_info = f"S3 Key: `{s3_key}` (no matching `modified_file_path` in `mst_documents`)"
                    source_info = "(File name not Found)"
            else:
                source_info = "No `s3_key` found in Pinecone metadata."

            response_parts.append(
                f"[Document Name:] {source_info}\n"
                # f"**Relevance Score:** {match['score']:.2f}\n"
                f"Content :\n{raw_text}"
            )

        # Combine all matched document snippets (between `[From The Document Content:]` and `[Document Name:]`)
        combined_response = "\n\n---\n\n".join(response_parts)

        return f"[From The Document Content:]\n\n{combined_response}"

    except Exception as e:
        logger.exception("pinecone_semantic_search failed")
        return "[From The Document Content:] Error during semantic search."

def pinecone_metadata_lookup(_: str, conv_id=None) -> str:
    logger.info(f"pinecone_metadata_lookup: conv_id={conv_id}")
    if not conv_id or conv_id not in PINECONE_LAST_MATCH:
        return "No Pinecone metadata found for this conversation. Please ask a document-related question first."
    meta = PINECONE_LAST_MATCH[conv_id]
    meta_lines = [f"{k}: {v}" for k, v in meta.items()]
    meta_str = "\n".join(meta_lines)
    return f"[Document Name:]\n{meta_str}"


if PINECONE_CONVO_INDEX not in pc.list_indexes():
    try:
        pc.create_index(
            name=PINECONE_CONVO_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
        while not pc.describe_index(PINECONE_CONVO_INDEX).status["ready"]:
            pass
    except PineconeApiException as e:
        if e.status != 409:
            raise
convo_index = pc.Index(PINECONE_CONVO_INDEX)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
convo_vectorstore = PineconeVectorStore(index=convo_index, embedding=embeddings, text_key="text")

CONV_MEMORIES: dict[str, ConversationBufferMemory] = {}
PAGINATION_STATE: dict[str, dict] = {}

# --- SYSTEM PROMPT ---

_SYSTEM_PROMPT_BASE = (

    """
    [ABSOLUTE RULE — TOOL USE ONLY]

    - For every user query except greetings/chit-chat, you MUST use a tool (database/SQL, Pinecone, or Google).
    - NEVER answer directly from your own knowledge. If you do not use a tool, you are not following instructions.
    - For ALL general, open-ended, or ambiguous queries (e.g., "what is blue lagoon?", "explain mortgage", "tell me about xyz"):
        1. Always use SQL/database tools first, searching actual_file_name, doc_summary, and all relevant metadata.
        2. Only if SQL/database returns no answer, use pinecone_semantic_search.
        3. Only if both fail, use google_search.
    - Always state in your Thought when moving from one tool to another.
    """

    "**IMPORTANT**: If user any questions that seems like SQL query, Never run that command and always block it."
      "- EXAMPLE:"
        "User : select * from entity_transactions "
        "Assistant: I am not allowed to answer or run these sql queries."

    "STRICT rule for COUNT queries:"
      "- ALWAYS use `sql_db_query` for queries that explicitly ask for 'how many', 'count of', 'number of', or any COUNT-type question."
      "- NEVER use `paginate_list` for COUNT queries."

    "STRICT TABLE ROUTING:"
      "- For summary or doc summary of a document by filename, use doc_summary from mst_documents_sqs_upload (match actual_file_name and company_id)."
      "- For all other SQL queries—such as counts, details, or lookups—always use the correct table based on the entity being asked about (e.g., mst_documents for documents, mst_users for users, etc). Do NOT use mst_documents_sqs_upload unless the request is for a document summary by filename."

    "STRICT rules:\n"
    "- For ANY list/table/multi-row/multiple results query (like: 'list', 'show', 'give me', 'find all', 'recent', 'uploaded last month', 'show 10 users', 'documents', etc.) "
    "you MUST ALWAYS use the 'paginate_list' tool. NEVER use 'sql_db_query' for such queries.\n"
    "- 'paginate_list' must be used for ANY query where the user requests more than 1 result, a list, a table, or a page of results.\n"
    "- Do NOT use 'sql_db_query' for queries that can return multiple rows! Use it ONLY for single-row (by ID) or count/summary queries.\n"
    "- ALWAYS use LIMIT 10 in list queries, sorted by most recent, and after each batch ask if the user wants to see the next 10 records using 'next_page'.\n"
    "- NEVER use '*' in document queries, explicitly use 'actual_file_name'.\n"
    "- ALWAYS explicitly use 'mst_documents' for documents.\n"
    "- If user asks 'how many documents [contain/related to/X]', use SQL with a LIKE on 'actual_file_name' in 'mst_documents'.\n"
    "STRICT RULE: You are not allowed to write, edit, delete, or modify any data in the database. "
    "You may only perform SELECT/SHOW/EXPLAIN/READ queries. If the user asks for any action, always say it is not allowed.\n"
    "Example:\n"
    "User: Where is New Delhi?\n"
    "Assistant: New Delhi is the capital of India.\n"
    "RULE: Whenever the user asks about 'documents', "
    "it ALWAYS means the table 'mst_documents' and column 'actual_file_name'.\n"
    "NEVER use any other table or column names for documents unless explicitly specified.\n\n"
    "Example:\n"
    "User: List documents uploaded last month\n"
    "Assistant: (Invoke paginate_list with SQL explicitly selecting 'actual_file_name' from 'mst_documents')\n\n"
    "---- Actual conversation below ----\n"
    "Previous conversation:\n{chat_history}\n\n"
    "Use prior conversation to resolve pronouns. If unclear, ask for clarification.\n\n"
    "You are an intelligent agent using ReAct. When using a tool, follow:\n"
    "  Thought: <reason>\n"
    "  Action: <tool name>\n"
    "  Action Input: <input>\n"
    "  Observation: <result>\n"
    "...repeat as needed...\n"
    "Final Answer: <your answer>\n\n"
    "Available tools:\n"
    "1) sql_db_query   â€“ single-item or summary (counts, single row). NOT for multi-row lists. NEVER use this for list/table queries.\n"
    "2) paginate_list  â€“ for ANY list-type query: returns 10 rows only. Must include:\n"
    "   â€¢ The 10 results, numbered\n"
    "   â€¢ The total count\n"
    "   â€¢ 'Would you like to see the next 10 records?'\n"
    "3) next_page      â€“ returns the next 10 rows of your last list.\n"
    "4) search_anywhere â€“ find a standalone ID across tables.\n"
    "5) pinecone_semantic_search â€“ answer unstructured content queries via Pinecone semantic search.\n"
    "6) pinecone_metadata_lookup â€“ get Pinecone metadata for the last document Q&A.\n"
    "7) google_search  â€“ fallback web search.\n\n"
    "STRICT rules:\n"
    "- ALWAYS use paginate_list for document lists or any multi-row/table/list queries.\n"
    "- NEVER use '*' in document queries, explicitly use 'actual_file_name'.\n"
    "- ALWAYS explicitly use 'mst_documents' for documents.\n\n"
    "Strict rules for lists: ALWAYS LIMIT 10, sorted by most recent. AFTER each batch, ask if the user wants the next 10.\n"
    "\n"

    "SUPER-STRICT COLUMN ROUTING FOR SPECIAL IDS:\n"
    "- Whenever the user asks about IRMES, IRMES number, IRMES ID, or IREMS_NUMBER (case-insensitive), ALWAYS assume the column is named 'index_value'. Use this in WHERE clauses (e.g., WHERE index_value = ...).\n"
    "- Whenever the user asks about 'record id' or 'recordid' (case-insensitive), ALWAYS use the column 'document_seq_id' in WHERE clauses.\n"
    "- Never guess column names for these queries—only use the specified mappings above.\n"


    "================ SEMANTIC SEARCH TOOL ROUTING ================\n"
    "Use the pinecone_semantic_search tool for ANY query where the user:\n"
    "  - asks to explain, describe, analyze, or understand the contents/text of a document, report, or file\n"
    "  - uses words or phrases such as 'explain', 'analyze', 'describe', 'meaning', 'interpret', 'what do you understand by', 'insights', 'what does the document say', or 'in your words'\n"
    "  - asks for information or an answer 'from the document', 'from the content', or 'from within the text'\n"
    
    "If the query is about the meaning, explanation, or interpretation of content inside a document, ALWAYS use pinecone_semantic_search ”even if the exact phrase 'from the document' is not present.\n"
    "SUPER-STRICT SUMMARY ROUTING:\n"
    "- If the user asks for a 'summary', 'doc summary', or to 'summarize' a specific document and provides a filename (like .pdf, .docx, .doc, .xls, etc.):\n"
    "    - ALWAYS use the SQL tool to select 'doc_summary' from the 'mst_documents_sqs_upload' table, using the provided filename (and company_id).\n"
    "    - NEVER use Pinecone semantic search for this case.\n"
    "    - Example: If user says \"summarize 800000000_1044 D Payment Information Treasury Financial(12).pdf\" or \"give doc summary for abc.pdf\", you must do:\n"
    "      SELECT doc_summary FROM mst_documents_sqs_upload WHERE actual_file_name = :filename AND company_id = :cid\n"
    "- Only use Pinecone semantic search for questions about the meaning, interpretation, or insights from the content of a document, not for doc_summary.\n"
    "EXAMPLES:\n"
    "User: Explain the findings in the audit report.\n"
    "Assistant: (Use pinecone_semantic_search on the audit report)\n"
    "User: What do you understand by 'indemnity' in this agreement?\n"
    "Assistant: (Use pinecone_semantic_search for the agreement document)\n"
    "User: Summarize this invoice.\n"
    "Assistant: (Use pinecone_semantic_search on the invoice)\n"
    "User: What does the document say about vendor payments?\n"
    "Assistant: (Use pinecone_semantic_search for the relevant document)\n"
    "User: List documents uploaded last month\n"
    "Assistant: (Do NOT use semantic search; use paginate_list instead)\n"
    "User: How many contracts do we have?\n"
    "Assistant: (Do NOT use semantic search; use sql_db_query)\n"
    "If in doubt, and the user's question relates to the meaning or understanding of document content, use pinecone_semantic_search.\n"
    "For follow-up questions like 'Show me more details from the document,' use pinecone_metadata_lookup.\n"
    "================================================================\n"
    "\n[FOR PINECONE/SEMANTIC TOOL] Whenever the user asks about 'from the document', 'source info', or similar, use 'pinecone_metadata_lookup' to show metadata.\n"

    # ------------ YOUR EXISTING RULES CONTINUE BELOW ---------------\n"
    "RULE: When the user asks for document type, type breakdown, or category, for example., finance, transfer approvals, mortgage, health etc "
    "NEVER use file extensions (like pdf, word) for document type unless the user explicitly says so.\n" 
    """
    1)m.actual_file_name LIKE CONCAT('%', dt.document_type, '%') → checks if the filename contains the document type text.

    2)dt.document_type LIKE '1044%' → filters only document types starting with 1044.


    For example:
    - List of documents with document type 1044 D Payment Information Treasury Financial.
    SELECT actual_file_name from mst_documents where actual_file_name like '%1044 D Payment Information Treasury Financial%'

    - List all the documents under document types whose name starts with "1044"
    SELECT  m.actual_file_name FROM mst_documents AS m JOIN document_types AS dt ON m.company_id = dt.company_id WHERE m.actual_file_name LIKE CONCAT('%', dt.document_type, '%') AND dt.document_type LIKE '%1044%';
                                                   

    - List of documents with document type 214 Transfer Approvals.
    SELECT  m.actual_file_name FROM mst_documents AS m JOIN document_types AS dt ON m.company_id = dt.company_id WHERE m.actual_file_name LIKE CONCAT('%', dt.document_type, '%') AND dt.document_type LIKE '%214 Transfer Approvals%'; """
    "RULE: When a user asks for details or searches for a word/phrase, use `keyword_search` to search for partial matches in all tables/columns. If multiple matches found, summarize types and ask the user to specify which type they want.\n"

    # ---------- NEW STRICT ROUTING RULE FOR ID/FILENAME BELOW ----------\n"
    "STRICT ROUTING:\n"
    "- Only use the 'search_anywhere' tool if the user query is a pure numeric ID (e.g., '12345', or similar).\n"
    "- If the query contains a file name (like 'abc.pdf', 'report on sales', '800000000_1044 D Payment Information Treasury Financial(323).pdf'), a document name, or any phrase that is not just an ID, NEVER use 'search_anywhere'.\n"
    "- For such cases (file names, document names, etc.), use the SQL or paginate_list tools to get the details from the database (e.g., SELECT * FROM mst_documents WHERE actual_file_name = :filename).\n"
    "- If unsure whether the user input is an ID or a file/document name, ask the user for clarification before taking any action.\n"

    """
    STRICT OWNER-OF-DOCUMENT ROUTING:
    - If the user asks "who is the owner of this document", "document owner", "owner of file", "who uploaded", or similar, 
    ALWAYS join mst_documents.doc_owner to mst_users.user_id to get first_name and last_name of the owner.
    - NEVER return just user_id, always return the owner's first_name and last_name.
    - Only return the owner for the document(s) actually present in mst_documents and for the correct company_id.
    - Example: 
        User: Who is the owner of 800000000_1044 D Payment Information Treasury Financial(12).pdf?
        SQL:
        SELECT u.first_name, u.last_name FROM mst_documents d
        JOIN mst_users u ON d.doc_owner = u.user_id
        WHERE d.actual_file_name = '800000000_1044 D Payment Information Treasury Financial(12).pdf'
        AND d.company_id = <company_id>;
    - Never guess user or document info; always require the exact file name or unique identifier.

    **DOCUMENT SUMMARY COUNT**:
    - For any count related question based on doc summary, summary or document summary, ALWAYS refer to mst_documents_sqs_upload table.                                           
    "EXAMPLE: "                   
    User: How many documents having document summary available
    Response: SELECT COUNT(*)  FROM mst_documents_sqs_upload WHERE doc_summary IS NOT NULL;                                       
    User: How many documents have doc summary available
    Response: SELECT COUNT(*)  FROM mst_documents_sqs_upload WHERE doc_summary IS NOT NULL;


    
"""
)




from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Hybrid SQL + Pinecone Agent", version="1.0.0")

@app.middleware("http")
async def catch_all_non_200(request: Request, call_next):
    # execute the request
    response = await call_next(request)

    # if anything other than a successful 200 comes back...
    if response.status_code != 200:
        # grab or generate a conversation ID so your front‑end still has one
        conv_id = request.headers.get("X-Conversation-ID", "") or str(uuid.uuid4())
        # log it for your own debugging
        logger.warning(f"Overriding {response.status_code} → friendly error for conv {conv_id}")

        # return exactly the same shape your UI expects
        return JSONResponse(
            status_code=200,  # so your front‑end sees it as "OK"
            content={
                "answer": "There is some technical glitch, please try after few minutes.",
                "conversation_id": conv_id,
                "confidence_score": 0.0
            },
        )

    # otherwise, return the real 200 response
    return response



class ToolUseTracker(BaseCallbackHandler):
    def __init__(self):
        self.tool_used = False
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_used = True

class QueryRequest(BaseModel):
    conversation_id: Optional[str] = None
    company_id: int
    question: str
    userEmail: str

class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    confidence_score: float
    

from langchain.callbacks.base import BaseCallbackHandler

class ThoughtCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []

    def on_llm_new_token(self, token: str, **kwargs):
        # Collect LLM streaming output if desired (for Thoughts)
        pass

    def on_agent_action(self, action, **kwargs):
        # This is called when the agent decides on an action/tool to use
        thought = action.log  # The ReAct-style "Thought: ...\nAction: ...\nAction Input: ..."
        print("[AGENT STEP]", thought)  # Print live to console
        self.steps.append(thought)

    def on_tool_end(self, output, **kwargs):
        print("[OBSERVATION]", output)
        self.steps.append(f"Observation: {output}")



def add_to_memory(conv_id, question, answer):
    logger.debug(f"add_to_memory: conv_id={conv_id}, question={question}, answer={answer}")
    if conv_id not in CONV_MEMORIES:
        from langchain.memory import ConversationTokenBufferMemory

        CONV_MEMORIES[conv_id] = ConversationTokenBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            llm=ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY),
            max_token_limit=4000,  # adjust if needed
            return_messages=True,
        )

    memory = CONV_MEMORIES[conv_id]
    memory.save_context({"input": question}, {"output": answer})



# LLM_CROSS_COMPANY_PROMPT = """
# You are an AI assistant. The user is authenticated with company_id: {company_id} (which is also file_plan_id: {company_id}). Hence user should accesss with all documents under company_id: {company_id}
 
# RULES (STRICT ENFORCEMENT):
 
# 1. If the user requests or refers to ANY company, program office (PROGRAM OFFICE, or COMPANY_ID/file_plan_id) that does NOT match their authenticated company_id ({company_id}), DO NOT provide access.
# 2. You MUST validate the reference using BOTH company/program/office name and company_id/file_plan_id.
#    - If a company/program/office name is mentioned, check that its corresponding id is {company_id}.
#    - If a company_id or file_plan_id is mentioned, check that it exactly matches {company_id}.
#    - Any mismatch, even in name or id, must result in access denial.

# 3."IMPORTANT: Numbers or codes like IREMS, reference numbers, document numbers, etc. are not company IDs or program office IDs unless directly prefixed by the word 'company id', 'company', or 'program office'. Ignore all such numbers when determining if a request is cross-company."   
 
# EXAMPLES OF FORBIDDEN REQUESTS:
# - User asks for data from another company/program/office by name or id.
# - User says "show me documents for program office MFHD" (and MFHD is not their own company_id).
# - User says "give me files for company ABC Corp" (and their company_id is not ABC Corp).
# - User asks for data for more than one company/program/office.
 
# EXPECTED RESPONSES:
# - If the user refers ONLY to their own company (by name or id), or is ambiguous, respond ONLY: "NO".
# - If the user refers to a different company/program/office (by name or id), respond ONLY: "YES".
# - If the user tries to access data for another company, you MUST respond: "Access Denied: You are in a different program office or company. Data access is not permitted."
 
# GIVEN USER QUESTION:
# "{question}"
 
# Based on the above, does the user try to access data for a different company, program office, or organization (by name or id) than their own ({company_id})?
# """
# def validate_file_belongs_to_company(file_keyword, user_company_id, engine):
#     query = text("""
#         SELECT COUNT(*) FROM mst_documents 
#         WHERE actual_file_name LIKE :file_keyword AND company_id = :company_id
#     """)
#     params = {
#         "file_keyword": f"%{file_keyword}%",
#         "company_id": user_company_id
#     }
#     with engine.connect() as conn:
#         count = conn.execute(query, params).scalar()
#     return count > 0


import re
from sqlalchemy import text

def get_all_company_names(engine):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT LOWER(company_name), company_id FROM mst_company")).fetchall()
    return {row[0]: row[1] for row in rows}

# At app startup:
ALL_COMPANIES = get_all_company_names(engine)


def extract_company_name_from_question(question: str, all_companies: dict) -> str | None:
    q_lower = question.lower()
    for cname in all_companies:
        if cname in q_lower:
            return cname  # matched a valid company_name
    return None


def get_company_id_from_question(question: str, user_company_id: int, engine) -> int | None:
    # Step 1: Explicit company_id mention?
    match = re.search(r'company[\s_]?id\s*=?\s*(\d+)', question, re.IGNORECASE)
    if match:
        found_company_id = int(match.group(1))
        return found_company_id if found_company_id == user_company_id else -1

    # Step 2: Check if any company/program office name in the user's question
    cname = extract_company_name_from_question(question, ALL_COMPANIES)
    if cname:
        mapped_id = ALL_COMPANIES[cname]
        if mapped_id == user_company_id:
            return user_company_id  # ✅ Allow access if company name matches user
        else:
            return -1  # ❌ Cross-company

    # Step 3: No company name match → assume user's own company
    return user_company_id


ROUTING_PROMPT = """
Decide the best tool for the following user question:


- If the user wants facts, structured data, lists, counts, summarize, summary, or anything present as a database field, choose "SQL".
- If the user wants to understand, analyze, or interpret from unstructured documents, or asks open-ended questions, choose "Semantic".
- If both apply, choose "Both".
- If it's unclear, choose "Semantic".

User question: {question}
Answer with only one word: SQL, Semantic, or Both.
"""

def classify_question_route(question: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
    chain = prompt | llm
    try:
        response = chain.invoke({"question": question}).content.strip().lower()
        if response in ("sql", "semantic", "both"):
            return response
        return "semantic"
    except Exception as e:
        # fallback: assume semantic if error
        return "semantic"    
    

import re

def is_doc_summary_request(question: str) -> str | None:
    m = re.search(
        r"(summarize|summary|doc\s*summary|provide the summary).*?([A-Za-z0-9_\-\. ()]+?\.(pdf|docx?|xls[xm]?|ppt[xm]?))",
        question, re.IGNORECASE)
    if m:
        return m.group(2).strip()
    m = re.search(
        r'summary (of|for)\s*([A-Za-z0-9_\-\. ()]+?\.(pdf|docx?|xls[xm]?|ppt[xm]?))',
        question, re.IGNORECASE)
    if m:
        return m.group(2).strip()
    return None

def fetch_doc_summary(filename: str, company_id: int, engine) -> str | None:
    query = text("""
        SELECT doc_summary FROM mst_documents_sqs_upload
        WHERE actual_file_name = :filename AND file_plan_id = :file_plan_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"filename": filename, "file_plan_id": company_id}).fetchone()
        if row and row[0]:
            return row[0]
    return None


async def format_semantic_response(llm, raw_response: str) -> str:
    prompt = (
        "You are an assistant. Below is the raw output from our semantic search tool:\n\n"
        "```\n"
        f"{raw_response}\n"
        "```\n\n"
        "Please:\n"
        "1. Extract the answer text (between `[From The Document Content:]` and `[Document Name:]`) and present it as a clear, concise paragraph.\n"
        "2. Under a **Source Metadata** heading, list each metadata field (everything after `[Document Name:]`) as bullet points.\n"
        "3. Use Markdown formatting for headings and bullets.\n\n"
        "Return only the formatted answer."
    )
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    return response.content.strip()


@app.post("/docqa", response_model=QueryResponse)
async def docqa(
    req: QueryRequest,
    response: Response,
):
    logger.info(f"Received docqa request: {req.dict()}")

    if is_permission_question(req.question):
        conv_id = req.conversation_id or str(uuid.uuid4())
        response.headers["X-Conversation-ID"] = conv_id
        return QueryResponse(
            answer="You are not allowed to ask permissions related questions",
            conversation_id=conv_id,
            confidence_score=0.0
        )


        # --- 1) LLM‑based small‑talk detection ---
    # small_talk_chain = (
    #     ChatPromptTemplate
    #       .from_template(SMALL_TALK_CLASSIFICATION_PROMPT)
    #     | ChatOpenAI(
    #         model_name="gpt-4.1",
    #         temperature=0,
    #         openai_api_key=OPENAI_API_KEY
    #     )
    # )
    # is_small_talk = (
    #     small_talk_chain
    #       .invoke({"question": req.question})
    #       .content
    #       .strip()
    #       .upper() 
    #     == "YES"
    # )
    # if is_small_talk:
    #     conv_id = req.conversation_id or str(uuid.uuid4())
    #     response.headers["X-Conversation-ID"] = conv_id
    #     # You can tweak this reply as you like:
    #     return QueryResponse(
    #         answer=(
    #             "Hi there! I'm your AI assistant. "
    #             "I'm doing great—thanks for asking! "
    #             "How can I assist you today?"
    #         ),
    #         conversation_id=conv_id,
    #         confidence_score=1.0
    #     )


    # ---- 1. Download file request handling (LLM-powered) ----
    download_response = handle_download_request_llm(req.question, req.company_id, engine, OPENAI_API_KEY)
    if download_response:
        conv_id = req.conversation_id or str(uuid.uuid4())
        response.headers["X-Conversation-ID"] = conv_id
        add_to_memory(conv_id, req.question, download_response)
        logger.info("File download requested and link generated via LLM intent detection.")
        return QueryResponse(
        answer=download_response,
        conversation_id=conv_id,
        confidence_score=1.0
        )
    

        # ---- 2) LLM-based “search keyword in summaries” ----
    summary_kw = extract_summary_keyword_via_llm(req.question, OPENAI_API_KEY)
    if summary_kw:
        logger.info(f"LLM detected summary‑keyword search for: {summary_kw}")
        docs = search_summaries(summary_kw, req.company_id, engine)
        conv_id = req.conversation_id or str(uuid.uuid4())
        response.headers["X-Conversation-ID"] = conv_id

        if docs:
            answer = (
                f"The following documents contain “{summary_kw}” in their summary:\n\n"
                + "\n".join(f"- {fn}" for fn in docs)
            )
            add_to_memory(conv_id, req.question, answer)
            return QueryResponse(
                answer=answer,
                conversation_id=conv_id,
                confidence_score=1.0
            )
        else:
            # No summary matches, fallback to content/semantic search!
            # You can use Pinecone or your existing keyword_search for content
            content_results = keyword_search(summary_kw, req.company_id, engine, limit=3)
            if content_results:
                files = []
                for res in content_results:
                    if res["table"] == "mst_documents" and "actual_file_name" in res["rows"][0]:
                        files.extend(row["actual_file_name"] for row in res["rows"])
                if files:
                    answer = (
                        f"No documents found with “{summary_kw}” in their summary.\n"
                        f"However, found in document content:\n" +
                        "\n".join(f"- {fn}" for fn in files)
                    )
                    add_to_memory(conv_id, req.question, answer)
                    return QueryResponse(
                        answer=answer,
                        conversation_id=conv_id,
                        confidence_score=0.9
                    )
            # Optionally, you can add Pinecone semantic search fallback here
            pinecone_resp = pinecone_semantic_search(summary_kw, conv_id=conv_id)
            if "[From The Document Content:]" in pinecone_resp and "No relevant answer" not in pinecone_resp:
                # answer = (
                #     f"No documents found with “{summary_kw}” in their summary.\n"
                #     f"However, found a semantic match in document content:\n\n{pinecone_resp}"
                # )
                answer = (
                    f"YES!, I found in the document content:\n\n{pinecone_resp}"
                )
                add_to_memory(conv_id, req.question, answer)
                return QueryResponse(
                    answer=answer,
                    conversation_id=conv_id,
                    confidence_score=0.85
                )
            # If neither found
            answer = f"No documents found with “{summary_kw}” in either their summary or content."
            add_to_memory(conv_id, req.question, answer)
            return QueryResponse(
                answer=answer,
                conversation_id=conv_id,
                confidence_score=0.0
            )


    
    # ADD THIS BLOCK:
    summary_filename = is_doc_summary_request(req.question)
    if summary_filename:
        logger.info(f"Detected summary request for file: {summary_filename}")
        summary_text = fetch_doc_summary(summary_filename, req.company_id, engine)
        conv_id = req.conversation_id or str(uuid.uuid4())
        response.headers["X-Conversation-ID"] = conv_id
        if summary_text:
            answer = f"Summary for **{summary_filename}**:\n\n{summary_text}"
        else:
            answer = f"Sorry, I could not find a summary for file: {summary_filename}."
        add_to_memory(conv_id, req.question, answer)
        logger.info("File summary fetched from mst_documents_sqs_upload.")
        return QueryResponse(
        answer=answer,
        conversation_id=conv_id,
        confidence_score=1.0
        )
    



    # Always define these at the start
    tracker = ToolUseTracker()
    thought_handler = ThoughtCaptureHandler()
    
    if is_sql_command_via_llm(req.question):
        logger.warning("User tried SQL command in request; blocked.")
        return QueryResponse(
        answer="I am not allowed to answer or run SQL queries or commands.",
        conversation_id=req.conversation_id or str(uuid.uuid4()),
        confidence_score=0.0
        )
    conv_id = req.conversation_id or str(uuid.uuid4())
    response.headers["X-Conversation-ID"] = conv_id
    user_company_id = req.company_id
    question_company_id = get_company_id_from_question(req.question, req.company_id, engine)
    logger.debug(f"Conv ID: {conv_id}, company_id: {user_company_id}, question_company_id: {question_company_id}")

    # question_company_id = get_company_id_from_question(req.question, req.company_id)
    if question_company_id == -1:
        logger.warning("Blocked cross-company prompt attempt.")
        return QueryResponse(
            answer="You are not allowed to specify or query for other companies/program offices. Please rephrase your question without mentioning another company or program office.",
            conversation_id=conv_id,
            thoughts="\n".join(thought_handler.steps),
            confidence_score=0.0
        )


    # --- always create/retrieve memory for this conversation ---
    if conv_id not in CONV_MEMORIES:
        from langchain.memory import ConversationTokenBufferMemory

        CONV_MEMORIES[conv_id] = ConversationTokenBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            llm=ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY),
            max_token_limit=4000,  # adjust if needed
            return_messages=True,
        )

    memory = CONV_MEMORIES[conv_id]

    # -------- Pagination tools --------
    def paginate_list_fn(input_val):
        logger.info(f"paginate_list_fn called with input_val={input_val}")
        import re
        from sqlalchemy import inspect
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        inspector = inspect(engine)
        llm = ChatOpenAI(
            model_name="gpt-4.1",
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )

        state = PAGINATION_STATE.get(conv_id)
        logger.debug(f"Current PAGINATION_STATE for {conv_id}: {state}")

        # --- "Next page" handling ---
        if isinstance(input_val, str):
            # Intercept "next N", "show next N", "more N", etc.
            next_n_match = re.match(r"^(next|show next|more|show)\s+(\d+)", input_val.strip().lower())
            if next_n_match:
                n_requested = int(next_n_match.group(2))
                if not state:
                    logger.warning("No previous list in this conversation for pagination.")
                    return {"error": "There's no previous list in this conversation. Could you tell me again what list you'd like to see?"}
                state["page"] += 1  # Only ever advance one page!
                res = handle_pagination(
                    engine,
                    state["base_query"],
                    state["params"],
                    page=state["page"],
                    per_page=10
                )
                return {
                    "rows": res["rows"],
                    "total": res["total"],
                    "page": res["page"],
                    "per_page": res["per_page"],
                    "message": f"You requested {n_requested} records, but I can only show results in batches of 10. Here are the next 10 records. Would you like to see the next batch?"
                }

            # Classic next/more/next page (no number)
            if re.match(r"^(next|more|show next)\b", input_val.strip().lower()):
                if not state:
                    logger.warning("No previous list in this conversation for pagination.")
                    return {"error": "There's no previous list in this conversation. Could you tell me again what list you'd like to see?"}
                state["page"] += 1
                res = handle_pagination(
                    engine,
                    state["base_query"],
                    state["params"],
                    page=state["page"],
                    per_page=10
                )
                return {
                    "rows": res["rows"],
                    "total": res["total"],
                    "page": res["page"],
                    "per_page": res["per_page"],
                    "message": "Showing the next 10 records. Would you like to see more?"
                }


        # New list query (via LLM prompt)
        if isinstance(input_val, str):
            # Schema overview for all included tables
            schema_desc = "\n".join(
            f"- {t} ({', '.join([c['name'] for c in inspector.get_columns(t)])})"
            for t in inspector.get_table_names()
            if t in SEARCH_TABLES
        )
            prompt = ChatPromptTemplate.from_template("""
    You are an intelligent SQL assistant.

    RULES:
    -NEVER invent table or column names. Use only from the schema below.
    -If unsure, ask the user for clarification instead of guessing.
                                                      
    - Only use tables and columns present in this schema.
    - Never use SELECT *, always specify columns by name.
    - For 'documents', always use mst_documents.actual_file_name.
    - For multi-row/list queries, always omit ORDER BY and LIMIT.
    - For string matching, use LIKE. Never use ILIKE.
    - Never use tables not present in the schema below.
    - For 'documents', always use mst_documents.actual_file_name.
    - NEVER use any other table or column names for documents unless explicitly specified.    

    ** VERY IMPORTANT **:
    - fFor listing documents having summary, always select 'mst_documents_sqs_upload' table                                   
                                                      
    🚩 **ENHANCED SUMMARY-TO-FILENAME ROUTING (DO NOT APPLY TO COUNT QUERIES):**

        -IMPORTANT: If the user asks for a count, number, or "how many" documents or records related to a keyword (e.g., "How many documents about finance?"), ALWAYS directly use the mst_documents table with conditions on actual_file_name or keywords. NEVER join mst_documents_sqs_upload for COUNT queries.
            
         WHENEVER the user explicitly requests summaries or explanations of documents, or explicitly asks to summarize, describe, explain, or analyze content related to a keyword (including partial words, names, or regions), you MUST:
            1. Extract ONLY the meaningful keyword or key phrase from the user's request.
                - Ignore additional context words (like "region", "area", "district", etc.) unless explicitly necessary.
                - If the user says "Wilkinson region", extract only "Wilkinson".
                - If user has spelling errors or partial words, still extract the best possible meaningful substring.

            2. Always generate SQL with a flexible match (`LIKE '%keyword%'`) against `doc_summary` column of `mst_documents_sqs_upload`.

            3. Always JOIN `mst_documents_sqs_upload` with `mst_documents` on `document_id`.

            4. ALWAYS RETURN BOTH `actual_file_name` and `doc_summary`.

            - NEVER require an exact match or the complete phrase if partial matches are possible.


        - For ALL OTHER queries about documents:
            - "List documents about X"
            - "List documents related to X"
            - "Show files containing X"
            - "List files mentioning X"
            - "Documents with X"
            - or similar phrasing without explicit "summarize", "content", "explain":
            
        ALWAYS directly query ONLY the `actual_file_name` (and optionally `keywords`) column from the `mst_documents` table. NEVER use `mst_documents_sqs_upload` or `doc_summary`.

        - User: "List documents related to finance"
        SQL:
        SELECT actual_file_name FROM mst_documents
        WHERE actual_file_name LIKE '%finance%'

        - User: "Show documents containing mortgage"
        SQL:
        SELECT actual_file_name FROM mst_documents
        WHERE actual_file_name LIKE '%mortgage%'

        - User: "Files mentioning audit"
        SQL:
        SELECT actual_file_name FROM mst_documents
        WHERE actual_file_name LIKE '%audit%'

        ✅ EXPLICIT CONTENT QUERY EXAMPLES (use mst_documents_sqs_upload.doc_summary):

        - User: "Summarize documents about finance"
        SQL:
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%finance%'

        - User: "Explain documents discussing mortgage"
        SQL:
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%mortgage%'

        ❌ INCORRECT EXAMPLES (DO NOT DO THIS):
        - User: "List documents related to finance"
        WRONG SQL (Do NOT use doc_summary here):
        SELECT d.actual_file_name FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%finance%'
                                                      
        EXAMPLES (Clearly showing handling typos, partial words, or extra context):

        ✅ User: "Summarize documents for Wilkinson region"
        SQL:
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%Wilkinson%'

        ✅ User: "Explain documents related to Finacial aproval"
        (User typo: "Finacial aproval" instead of "Financial approval")
        SQL (flexible partial match):
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%Finac%' OR s.doc_summary LIKE '%aprov%'

        ✅ User: "Summarize files about Morgage reports"
        (User typo: "Morgage" instead of "Mortgage")
        SQL (flexible partial match):
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%Morga%'


        - **Never return document_id or any internal IDs to the user—only actual_file_name (and optionally doc_summary if requested).**


        🚩 CLEAR HANDLING OF AMBIGUOUS QUERIES (BOTH FILENAME & SUMMARY):

        - If the user's request is ambiguous or unclear whether they want filenames or summaries (e.g., "show documents about X", "documents involving X", "give information on X documents"), you MUST clearly and explicitly perform TWO separate SQL searches:

        1. Search filenames (`actual_file_name`):
            SELECT actual_file_name FROM mst_documents
            WHERE actual_file_name LIKE '%<keyword>%'

        2. Search summaries (`doc_summary`):
            SELECT d.actual_file_name, s.doc_summary
            FROM mst_documents_sqs_upload s
            JOIN mst_documents d ON s.document_id = d.document_id
            WHERE s.doc_summary LIKE '%<keyword>%'

        - Clearly present both results separately to the user:

        EXAMPLES OF CLEARLY HANDLING AMBIGUOUS REQUESTS:

        ✅ User: "Show documents about Blue Lagoon"

        Filename matches:
        SELECT actual_file_name FROM mst_documents
        WHERE actual_file_name LIKE '%Blue Lagoon%'

        Summary matches:
        SELECT d.actual_file_name, s.doc_summary
        FROM mst_documents_sqs_upload s
        JOIN mst_documents d ON s.document_id = d.document_id
        WHERE s.doc_summary LIKE '%Blue Lagoon%'

        Then clearly respond to the user:

        Filename Matches:
        - Blue Lagoon Financial Report 2024.pdf
        - Audit of Blue Lagoon Payments.docx
        
        Summary Matches:
        - Document: Vendor Contract Q3.pdf
        Summary: This document covers transactions made at Blue Lagoon during Q3.


        
                                                      
    "SUPER-STRICT COLUMN ROUTING FOR SPECIAL IDS:\n"
    "- Whenever the user asks about IRMES, IRMES number, IRMES ID, or IREMS_NUMBER (case-insensitive), ALWAYS assume the column is named 'index_value'. Use this in WHERE clauses (e.g., WHERE index_value = ...).\n"
    "- Whenever the user asks about 'record id' or 'recordid' (case-insensitive), ALWAYS use the column 'document_seq_id' in WHERE clauses from the mst_documents.\n"                                                                                                                                                         
    "- Never guess column names for these queries—only use the specified mappings above.\n"
    - The IRMES number (index_value) exists ONLY in the doc_index_mapping table, NOT in mst_documents.
    - To list document file names for a given IRMES number, JOIN mst_documents and doc_index_mapping ON document_id.
    - Whenever the user asks about the keyword such as 'blue lagoon', 'payment transfer', 'transfer approval', 'finance', 'payment', 'mortgage', etc , ALWAYS consider actual_file_name column in mst_documents table.
    - Whenever the user ask about username such as 	C62094, etc, or something related to this user name, ALWAYS consider creted_By column of mst_documents table.
    - Whenever the user ask about documents under any particular document type, ALWAYS consider actual_file_name in mst_documents table with that company_id.
    - Whenever the user ask about documents having keyword or containing keyword or 
                                                      
    - Company ID Validation Rule:
    - If the user asks about a specific company_id, ALWAYS compare it with the company_id in the request body.
        - If both match, generate the SQL as usual using the company_id from the request body.
        - If they do not match, DO NOT generate SQL. Respond with: "The company_id in your request does not match your session. Please provide the correct company_id."
    - Never guess, infer, or reveal data for unmatched company_ids.
    
                                                      
    - Example:
    SELECT d.actual_file_name
    FROM mst_documents d
    JOIN doc_index_mapping m ON m.document_id = d.document_id
    WHERE m.index_value = '800090909';
                                                       
    - Example:
    "Give me details of report id 8991179"
    SELECT * FROM mst_documents WHERE document_seq_id = 8991179                                                                                                                                                  

    Examples:
    - List documents from last month:
    SELECT actual_file_name FROM mst_documents WHERE upload_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);

    - List all users:
    SELECT user_id, full_name, email_id FROM mst_users;

    - List entity payments this year:
    SELECT payment_id, payment_amount, payment_date FROM entity_payments WHERE payment_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR);
                                                      
    - List of documents containing the word blue lagoon in their file name.
    SELECT actual_file_name from mst_documents where actual_file_name like '%blue lagoon%'
                                                      
    - List of documents with document type 1044 D Payment Information Treasury Financial.
    SELECT actual_file_name from mst_documents where actual_file_name like '%1044 D Payment Information Treasury Financial%'

    - List all the documents under document types whose name starts with "1044"
    SELECT m.actual_file_name FROM mst_documents AS m JOIN document_types AS dt ON m.company_id = dt.company_id WHERE m.actual_file_name LIKE CONCAT('%', dt.document_type, '%') AND dt.document_type LIKE '%1044%';
                                                   

    - List of documents with document type 214 Transfer Approvals.
    SELECT  m.actual_file_name FROM mst_documents AS m JOIN document_types AS dt ON m.company_id = dt.company_id WHERE m.actual_file_name LIKE CONCAT('%', dt.document_type, '%') AND dt.document_type LIKE '%214 Transfer Approvals%';                                                                                                  
                                                      

    - List all the documents related to payment.
    SELECT actual_file_name from mst_documents where actual_file_name like '%payment%' ORDER BY upload_date DESC LIMIT :limit OFFSET :offset

    - List of documents created by user C62094 .
    SELECT actual_file_name from mst_documents where created_By = 'C62094' 
                                                                                                                                                                                                      

    - User question: "List all documents with company_id 9."
    - Request body: company_id = 9
    - Response:
        SELECT actual_file_name FROM mst_documents WHERE company_id = 9

    - User question: "List all documents with company_id 9."
    - Request body: company_id = 11
    - Response:
        The company_id in your request does not match your session. Please provide the correct company_id.

    - User question: "List entity payments for company_id 5."
    - Request body: company_id = 5
    - Response:
        SELECT payment_id, payment_amount, payment_date FROM entity_payments WHERE company_id = 5

    - User question: "Show documents for company_id 12."
    - Request body: company_id = 10
    - Response:
        The company_id in your request does not match your session. Please provide the correct company_id.    
                                                      
    **STRICT REGION/PROPERTY QUERY ROUTING:(PAGINATED)**
    - For any “list” or “show” request filtered by region, state, city, or property attribute, you must:
        1. JOIN `mst_documents` to `doc_index_mapping` ON `document_id = document_id`.
        2. JOIN  to `mst_property` ON `index_value = property_id` AND `company_id = company_id`.
        3. Add your filter in `WHERE` (e.g. `p.state = 'WI'`).
        4. ALWAYS show the results in the pagination of LIMIT 10 only and user says next, then show next 10 records.

        **Example for REGION/PROPERTY QUERY ROUTING:**
        User: “List all documents from the Wisconsin region.”  
        reponse: select d.* FROM mst_documents d JOIN doc_index_mapping dm on d.document_id = dm.document_id JOIN mst_property p ON d.company_id = p.company_id and dm.index_value = p.property_id WHERE p.property_name like '%Wisconsin%';          

        
        **Whenever user asks about list of doc summary or document summary, ALWAYS refer to mst_documents_sqs_upload table.**

        EXAMPLE:
        User: List all the documents having doc summary avaiable  
        Response: SELECT s.actual_file_name FROM mst_documents_sqs_upload s WHERE s.doc_summary IS NOT NULL
                                                                 
                

    User asked: "{query}"

    Schema:
    {schema}

    Respond ONLY with the SQL query, with NO markdown fences, and don't add ORDER BY, LIMIT or OFFSET.
    """)
            chain = prompt | llm
            try:
                raw_sql = chain.invoke({
                        "query": input_val,
                        "schema": schema_desc,
                        # "req.company_id":  req.company_id  # <-- Make sure this variable is available here
                    }).content.strip()
                logger.debug(f"LLM generated raw SQL: {raw_sql}")
                sql_query = re.sub(r"```(?:sql)?\n?", "", raw_sql, flags=re.IGNORECASE)
                sql_query = sql_query.replace("```", "").replace("`", "").strip().rstrip(";")
            except Exception as e:
                logger.exception("Error generating SQL from LLM in paginate_list_fn")
                return {"error": str(e)}

            # Patch: fix obvious LLM errors
            # Patch table names, force only included tables
            table_match = re.search(r'from\s+([`"]?)(\w+)\1', sql_query, re.IGNORECASE)
            if table_match:
                tname = table_match.group(2)
                if tname not in SEARCH_TABLES:
                    logger.warning(f"Invalid table used in SQL: {tname}")
                    return {"error": f"You can only use allowed tables: {', '.join(SEARCH_TABLES)}"}
            # Patch invalid columns
            # Patch invalid columns (alias‑aware)
            cols = [c['name'] for c in inspector.get_columns(tname)]
            raw_select = re.findall(r'select\s+(.*?)\s+from', sql_query, re.IGNORECASE)[0]
            select_cols = [c.strip() for c in raw_select.split(',')]

            invalid_cols = []
            for raw in select_cols:
                # strip off any "alias."
                col_name = raw.split('.')[-1]
                if col_name not in cols:
                    invalid_cols.append(raw)

            if invalid_cols:
                logger.warning(f"Invalid columns in SQL: {invalid_cols}")
                return {
                    "error": f"Invalid columns in query: {', '.join(invalid_cols)}. "
                            f"Allowed columns: {', '.join(cols)}"
                }

                
    

            # Fix SELECT * (force explicit columns)
            if re.search(r"select\s+\*\s+", sql_query, re.IGNORECASE):
                # Find table, get its columns, replace * with those
                if table_match:
                    tname = table_match.group(2)
                    cols = [c['name'] for c in inspector.get_columns(tname)]
                    sql_query = re.sub(
                        r"select\s+\*\s+from",
                        f"select {', '.join(cols)} from",
                        sql_query,
                        flags=re.IGNORECASE
                    )
                else:
                    return {"error": "Query used SELECT * but table was not detected. Please rephrase your question."}

            # Enforce company_id filter if present
            sql_query = inject_company_id_filter(sql_query, req.company_id, engine)
            logger.info(f"Final SQL for pagination: {sql_query}")
            if not is_readonly_sql(sql_query):
                logger.error("paginate_list_fn: Non-readonly query detected.")
                return {"error": "Sorry, I am only allowed to read data. No changes or actions can be made in the database."}

            PAGINATION_STATE[conv_id] = {
                "base_query": sql_query,
                "params": {},
                "page": 1,
                "per_page": 10
            }
            res = handle_pagination(engine, sql_query, {}, page=1, per_page=10)

            # 1) if handle_pagination returned an error, bubble it up
            if "error" in res:
                logger.error(f"paginate_list_fn: underlying error: {res['error']}")
                return {"error": res["error"]}

            # 2) guard against missing or empty rows
            rows = res.get("rows", [])
            if not rows:
                logger.info("paginate_list_fn: No rows found for query.")
                return {
                    "rows": [],
                    "total": 0,
                    "page": 1,
                    "per_page": 10,
                    "message": "this information is not available in the current program office"
                }

            # 3) otherwise, return normally
            return {
                "rows": rows,
                "total": res.get("total", 0),
                "page": res.get("page", 1),
                "per_page": res.get("per_page", 10),
                "message": ""
            }


        # Tuple/list/dict fallback (for internal)
        if isinstance(input_val, (list, tuple)):
            bq = input_val[0] if len(input_val) > 0 else ""
            prm = input_val[1] if len(input_val) > 1 else {}
            pg = input_val[2] if len(input_val) > 2 else 1
            pp = 10
            if not is_readonly_sql(bq):
                return {"error": "Sorry, I am only allowed to read data. No changes or actions can be made in the database."}
            r = handle_pagination(engine, bq, prm, page=pg, per_page=pp)
            return {"rows": r["rows"], "total": r["total"], "page": r["page"], "per_page": r["per_page"], "message": ""}

        if isinstance(input_val, dict):
            bq = input_val.get("base_query", "")
            prm = input_val.get("params", {})
            pg = input_val.get("page", 1)
            pp = 10
            if not is_readonly_sql(bq):
                return {"error": "Sorry, I am only allowed to read data. No changes or actions can be made in the database."}
            r = handle_pagination(engine, bq, prm, page=pg, per_page=pp)
            return {"rows": r["rows"], "total": r["total"], "page": r["page"], "per_page": r["per_page"], "message": ""}

        return {"error": "Invalid input format for paginate_list"}



    def next_page_tool_fn(_: str):
        logger.info(f"next_page_tool_fn called for {conv_id}")
        logger.debug(f"PAGINATION_STATE for {conv_id}: {PAGINATION_STATE.get(conv_id)}")
        print(f"DEBUG: PAGINATION_STATE for {conv_id}: {PAGINATION_STATE.get(conv_id)}")
        return paginate_list_fn("next")


    def is_probably_id(q):
    # Only allow search_anywhere for pure IDs (all digits, no spaces, no .pdf, etc)
        return (
            q.strip().isdigit() or
            (len(q.strip()) < 30 and re.match(r"^\d+$", q.strip()))
        )

    def search_anywhere_tool_fn(q):
        logger.info(f"search_anywhere_tool_fn called with query: {q}")
        if not is_probably_id(q):
            return (
                "The ID search tool can only be used with a *pure ID number* (e.g., '12345'). "
                "For names, filenames, or detailed queries, use another tool."
            )
        return search_anywhere(q, req.company_id)

    search_anywhere_tool = Tool.from_function(
        func=search_anywhere_tool_fn,
        name="search_anywhere",
        description="Locate a single record by its numeric ID (e.g., 12345). Use ONLY for questions that are pure IDs. DO NOT use this for filenames or names."
    )


    paginate_tool = Tool.from_function(
        func=paginate_list_fn,
        name="paginate_list",
        description= "Use this tool for ANY list/table query, or anything returning multiple results. "
        "Examples: 'list documents', 'show 10 documents', 'recent uploads', 'all users', etc. "
        "NEVER use sql_db_query for these queries.",
    )
    next_page_tool = Tool.from_function(
        func=next_page_tool_fn,
        name="next_page",
        description="Fetch the next 10 rows of your last list request.",
    )

    # # --- KEYWORD SEARCH TOOL ---
    # def keyword_search_tool_fn(query: str):
    #     results = keyword_search(query, req.company_id, engine)
    #     return handle_keyword_results(results)
    # keyword_search_tool = Tool.from_function(
    #     func=keyword_search_tool_fn,
    #     name="keyword_search",
    #     description="Search all tables for records matching the given keyword or phrase in any string field. Returns possible matches and asks the user to clarify if there are multiple types."
    # )

    # ------- STRICT SEMANTIC ROUTING ONLY IF "from the document" -----
    def semantic_tool_wrapper(question):
    # Always allow, LLM routing decides when to call
        return pinecone_semantic_search(question, conv_id=conv_id)


    semantic_tool = Tool.from_function(
        func=semantic_tool_wrapper,
        name="pinecone_semantic_search",
        description="Use this ONLY after you have tried SQL/database tools and found no answer. "
        "Do NOT use for general or ambiguous questions until SQL/database is attempted."
    )

    def pinecone_meta_tool_wrapper(_: str):
        return pinecone_metadata_lookup(_, conv_id=conv_id)

    pinecone_meta_tool = Tool.from_function(
        func=pinecone_meta_tool_wrapper,
        name="pinecone_metadata_lookup",
        description="Get Pinecone metadata for the last document Q&A in this conversation."
    )

    custom_sql_tools = []
    
    for t in sql_tools:
        if t.name == "sql_db_query":
            # Re-wrap the tool to inject company_id AND stricter description
            sql_db_tool = make_company_id_sql_tool(t, req.company_id, engine)
            # Overwrite the description to strictly forbid use for lists
            sql_db_tool.description = ('''
                "Use this for any database or structured information lookup, and always as the first step for any generic question."
                "Use this tool ONLY for:\n"
                "- Single-row lookup by ID (e.g. details for a specific document, user, property by ID) in the appropriate table.\n"
                "- Count/summary queries (e.g. 'how many', 'number of', 'count of') in the appropriate table for the entity (e.g. documents in mst_documents, users in mst_users, etc.).\n"
                "- For summary/doc summary requests for a specific document by filename (such as .pdf, .docx, etc), ONLY THEN select doc_summary from mst_documents_sqs_upload WHERE actual_file_name = :filename AND company_id = :cid.\n"
                "IMPORTANT: For count, number, or any other query, ALWAYS use the correct table as per the question context and always validate query is generated is having matching company_id column value with reponse company_id = {req.company_id}. Do NOT use mst_documents_sqs_upload unless the user explicitly asks for a summary of a specific file by filename.\n"
                "NEVER use this tool for listing tables or for queries that return multiple rows. For lists, use paginate_list.\n"
                 
                "- When user asks anything related to user name, ALWAYS select the created_By column in mst_documents table."
                
                Document Type examples:
                question - Count of all documents under document types starting with 1044
                SELECT COUNT(m.actual_file_name) AS total_docs
                FROM mst_documents AS m
                JOIN document_types AS dt
                    ON m.company_id = dt.company_id
                WHERE m.actual_file_name LIKE CONCAT('%', dt.document_type, '%')
                AND dt.document_type LIKE '%1044%';                      
                
                **IMPORTANT**
                " When the user asks about a "record", "record id", or "record_id", always treat this as referring to the document_seq_id in the mst_documents table.
                    Only fetch and return these columns: actual_file_name, created_Date, upload_date, and retirement_date.
                                       
                "\nSTRICT OWNER QUERY ROUTING:"
                "\n- For 'owner of this document', always join mst_documents.doc_owner to mst_users.user_id and select first_name, last_name."
                "\n- Example: SELECT u.first_name, u.last_name FROM mst_documents d JOIN mst_users u ON d.doc_owner = u.user_id WHERE d.actual_file_name = '<filename>' AND d.company_id = <company_id>;"
                 
                                       
                **STRICT REGION/PROPERTY QUERY ROUTING:**
                - For any “list” or “show” request filtered by region, state, city, or property attribute, you must use paginate_list_fn
                                      
                "Example: "
                  "-If the user asks any property related question, ALWAYS consider mst_property table." 
                  "-If the user asks any user name related question, ALWAYS consider created_By column from mst_documents table. "
                                       
                **DOCUMENT SUMMARY COUNT**:
                - For any count related question based on doc summary, summary or document summary, ALWAYS refer to mst_documents_sqs_upload table.                                           
                "EXAMPLE: "                   
                User: How many documents having document summary available
                Response: SELECT COUNT(*)  FROM mst_documents_sqs_upload WHERE doc_summary IS NOT NULL;                                       
                User: How many documents have doc summary available
                Response: SELECT COUNT(*)  FROM mst_documents_sqs_upload WHERE doc_summary IS NOT NULL;
                                       

               
            ''')
            custom_sql_tools.append(sql_db_tool)
        else:
            # Optionally, you can make all other tool descriptions stricter if you wish
            custom_sql_tools.append(t)


    tools = [
        search_anywhere_tool,
        #keyword_search_tool,     # <-- Keyword search included
        *custom_sql_tools,
        paginate_tool,
        next_page_tool,
        semantic_tool,
        pinecone_meta_tool,
    ]

    should_use_convo_vectorstore = "from the document" in req.question.lower()
    system_message = _SYSTEM_PROMPT_BASE + f"\n\n[HIDDEN] company_id = {req.company_id}"
    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY),
        agent=AgentType.OPENAI_FUNCTIONS,
        system_message=system_message,
        memory=memory,
        input_variables=["input", "chat_history"],
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=20,
        early_stopping_method="force",
        function_call="auto"
    )

    

    # 1) Build up the “augmented_prompt” exactly as you had before:
    history = "\n".join(
        f"user: {m.content}" if m.type == "human" else f"assistant: {m.content}"
        for m in getattr(memory, "buffer", []) or []
    )
    prompt = (
        f"Previous conversation:\n{history}\n\n"
        f"Now the new question is:\n{req.question}"
    )
    augmented_prompt = prompt
    if "from the document" in req.question.lower():
        try:
            results = convo_vectorstore.similarity_search(req.question, k=3)
            mems = [doc.page_content for doc in results]
            if mems:
                augmented_prompt = (
                    "Relevant previous Q&As:\n"
                    + "\n".join(mems)
                    + "\n\n"
                    + prompt
                )
        except Exception:
            logger.exception("Failed to retrieve from convo_vectorstore")


        # ─── SQL‑first with fallback to Semantic ───
    import random

    logger.info("Attempting SQL agent first for question: %s", req.question)
    sql_answer = await agent.arun(augmented_prompt, callbacks=[tracker, thought_handler])

    def is_empty_sql(ans: str) -> bool:
        txt = ans.strip().lower()
        return (
            not txt
            or txt.startswith("no rows")
            or "no results" in txt
            or "not available in the current program office" in txt
        )

    if not is_empty_sql(sql_answer):
        # SQL returned something useful
        answer = sql_answer
        confidence = round(random.uniform(0.95, 1.0), 3)
        logger.info("SQL agent returned data; skipping semantic.")
    else:
        # SQL came up empty → fallback to Pinecone
        logger.info("SQL agent empty; falling back to semantic search.")
        raw = pinecone_semantic_search(req.question, conv_id=conv_id)
        answer = await format_semantic_response(
            ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
            ),
            raw
        )
        confidence = PINECONE_LAST_SCORE.get(conv_id, 0.0)

    # Save into memory & convo store
    add_to_memory(conv_id, req.question, answer)
    try:
        convo_vectorstore.add_texts([f"Q: {req.question}\nA: {answer}"])
    except Exception:
        logger.exception("Failed to add to convo_vectorstore")

    # And finally return
    return QueryResponse(
        answer=answer,
        conversation_id=conv_id,
        confidence_score=confidence
    )
