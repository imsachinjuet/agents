import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    """
    Extract summary keywords from a question using LLM
    
    Args:
        question: The user's question
        
    Returns:
        LLM response indicating if it's a summary keyword search
    """
    prompt = ChatPromptTemplate.from_template(SUMMARY_KEYWORD_PROMPT)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    resp = (prompt | llm).invoke({"question": question}).content.strip()
    return resp

def main():
    """Main function to test the LLM functionality"""
    print("🤖 LangChain Agent - Summary Keyword Extraction")
    print("=" * 50)
    
    # Test the LLM function
    test_questions = [
        "what is sum of 2 and 2",
        "search for documents about machine learning",
        "find summaries containing python programming",
        "what is the weather today"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = extract_summary_keyword_via_llm(question)
        print(f"Result: {result}")
    
    print("\n" + "=" * 50)
    print("✅ Agent1.py is now clean and focused on LangChain functionality!")
    print("💡 For Supabase operations, use the SupabaseClient class from supabase_client.py")

if __name__ == "__main__":
    main()