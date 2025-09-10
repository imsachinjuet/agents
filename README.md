te .venv# Supabase + LangChain Integration

This project demonstrates how to integrate Supabase with LangChain for building AI-powered applications with database storage, vector search, and document management capabilities.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   python setup.py
   ```

2. **Configure environment:**
   - Update `.env` file with your OpenAI API key
   - Supabase credentials are already configured

3. **Run the main agent:**
   ```bash
   python agent1.py
   ```

## 📁 Project Structure

- `agent1.py` - Main agent with Supabase integration and SQL toolkit
- `supabase_langchain_example.py` - Vector store and RAG pipeline examples
- `setup.py` - Setup script for dependencies and configuration
- `.env` - Environment configuration (update with your API keys)
- `requirements.txt` - Python dependencies

## 🔧 Configuration

### Supabase Settings
Your Supabase project is already configured:
- **URL:** https://lbchkzuwikzapdkgeeju.supabase.co
- **API Key:** Configured in .env file

### Environment Variables
Update `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

For SQL database features, also add:
```env
SUPABASE_DB_URI=postgresql://postgres:YOUR_DB_PASSWORD@db.lbchkzuwikzapdkgeeju.supabase.co:5432/postgres
```

## 🎯 Features

### 1. Basic Supabase Integration
- Direct API client connection
- Table operations (CRUD)
- Real-time subscriptions support

### 2. LangChain SQL Agent
- SQL database toolkit integration
- Natural language to SQL queries
- Automatic schema introspection

### 3. Vector Store Integration
- Supabase as LangChain vector store
- Document embedding and storage
- Similarity search capabilities

### 4. RAG Pipeline
- Retrieval-Augmented Generation
- Document chunking and indexing
- Context-aware responses

## 📊 Database Setup

### For Vector Search (Optional)
Create a documents table in your Supabase dashboard:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);

-- Create index for faster similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create match function for similarity search
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.78,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

## 🔍 Usage Examples

### Basic Supabase Operations
```python
from supabase import create_client

# Insert data
supabase.table('your_table').insert({'column': 'value'}).execute()

# Query data
result = supabase.table('your_table').select('*').execute()

# Update data
supabase.table('your_table').update({'column': 'new_value'}).eq('id', 1).execute()
```

### Vector Search
```python
from langchain_community.vectorstores import SupabaseVectorStore

# Setup vector store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents"
)

# Add documents
vector_store.add_documents(documents)

# Search similar documents
results = vector_store.similarity_search("your query", k=5)
```

### SQL Agent
```python
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Create SQL agent
agent = initialize_agent(
    toolkit.get_tools(),
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Natural language query
response = agent.run("Show me all users created in the last week")
```

## 🔒 Security Notes

- API keys are stored in `.env` file (gitignored)
- Use Row Level Security (RLS) in Supabase for production
- Implement proper authentication for sensitive operations

## 🐛 Troubleshooting

### Common Issues

1. **Connection Error:**
   - Verify Supabase URL and API key
   - Check network connectivity

2. **SQL Features Not Working:**
   - Add database password to `SUPABASE_DB_URI`
   - Ensure database is accessible

3. **Vector Search Issues:**
   - Install pgvector extension in Supabase
   - Create proper table schema
   - Verify embedding dimensions match

## 📚 Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
