import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate

# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

# Prompt template
SUMMARY_KEYWORD_PROMPT = """
You are an AI assistant. Determine whether the user is asking to search for a keyword inside existing document summaries.

DO NOT consider any IREMS number or IREMS or anything related to IREMS inside this summary.

- If yes, reply exactly with:
  SEARCH_SUMMARY_KEYWORD:{{keyword}}

- Otherwise reply with exactly:
  NO

User message: {question}
"""

def extract_summary_keyword_via_llm(question: str) -> str | None:
    prompt = ChatPromptTemplate.from_template(SUMMARY_KEYWORD_PROMPT)
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)
    resp = (prompt | llm).invoke({"question": question}).content.strip()
    return resp

print(extract_summary_keyword_via_llm("what is sum of 2 and 2"))

# Database
db = SQLDatabase.from_uri(SUPABASE_DB_URI)

print("Tables:", db.get_usable_table_names())

# SQL Toolkit
llm2 = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=OPENAI_API_KEY)
toolkit = SQLDatabaseToolkit(db=db, llm=llm2)

# Agent
agent = initialize_agent(
    toolkit.get_tools(),
    llm2,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("List all tables in the Supabase database.")
print(response)