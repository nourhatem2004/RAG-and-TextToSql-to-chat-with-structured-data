# Structured Data Retrieval: RAG & Text-to-SQL Chatbot

This project is an AI-powered system for querying structured databases using natural language. It combines Retrieval-Augmented Generation (RAG), semantic search, and generative models to answer user questions with SQL queries and friendly explanations.

## Features

- **Natural Language to SQL**: Converts user questions into SQL queries using Google Gemini and HuggingFace embeddings.
- **Database Schema Awareness**: Extracts and embeds schema info for context-aware query generation.
- **Semantic Search**: Uses Qdrant vector database for fast retrieval of relevant schema.
- **Conversational Responses**: Returns both query results and a human-friendly explanation of the answer.

## How It Works

1. **Schema Extraction**: Connects to a SQL Server database and extracts schema details (tables, columns, datatypes).
2. **Embedding & Storage**: Embeds schema info using HuggingFace, stores them in Qdrant collections.
3. **Keyword Extraction**: Uses Gemini to extract 5 relevant keywords from the user’s question.
4. **Context Retrieval**: Searches Qdrant for schema and example contexts matching the keywords.
5. **SQL Generation**: Prompts Gemini to generate an appropriate SQL query and explain its logic.
6. **Query Execution**: Runs the generated SQL against the database and retrieves results.
7. **Chat Response**: Uses Gemini to generate a friendly, context-aware answer for the user.

## Requirements

- Python 3.11+
- SQL Server (I use AdventureWorks2022 database, you can work with any database)
- [ODBC Driver 17 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
- Packages:
  - `python-dotenv`
  - `google-generativeai`
  - `pyodbc`
  - `langchain_huggingface`
  - `qdrant-client`
  - `numpy`
  - (See requirments.txt for full list)

## Setup

1. **Clone the repository** and navigate to the project folder.
2. **Create a `.env` file** with your Google API key:
	```
	GOOGLE_API_KEY=your_google_api_key_here
	```
3. **Activate the virtual environment**:
	```
	.\structured_env\Scripts\Activate.ps1
	```
4. **Install dependencies** (if not already installed):
	```
	pip install -r requirements.txt
	```
5. **Configure SQL Server**:
	- Ensure SQL Server is running locally.
	- Your Example database is available.

## Usage

Run the main script:

```powershell
python main.py
```

The script will:
- Connect to the database
- Extract and embed schema info
- Accept a sample question (edit in code for custom questions)
- Generate SQL, execute it, and print a friendly answer

## Technical Details

### Embeddings & Semantic Search
- **HuggingFace Embeddings**: The system uses the `all-MiniLM-L6-v2` model to convert text (schema descriptions, questions) into dense vector representations. This enables semantic similarity search, allowing the system to match user queries with relevant schema elements and example SQL pairs.

### Vector Database: Qdrant
- **Qdrant** is used as the vector database for storing and searching embeddings. Collections are created for both schema info and example SQL pairs, each configured for cosine similarity and on-disk storage.
- **Why Qdrant over FAISS?**
  - **Production-Ready**: Qdrant is a full-featured vector database with REST/gRPC APIs, persistent storage, and advanced filtering. FAISS is primarily a library for in-memory similarity search and lacks database features.
  - **Scalability**: Qdrant supports sharding, replication, and disk-based storage, making it suitable for large-scale, persistent deployments. FAISS is best for research or small-scale, in-memory use cases.
  - **Filtering & Metadata**: Qdrant allows filtering search results by metadata (payload), which is essential for context-aware retrieval (e.g., filtering by table, column, etc.). FAISS does not natively support this.
  - **Ease of Integration**: Qdrant has Python, REST, and other client libraries, making integration straightforward for modern AI applications.
- **Usage in this project**:
  - Schema and example SQL pairs are embedded and stored as points in Qdrant collections.
  - For each user question, extracted keywords are embedded and used to search Qdrant for relevant schema and example contexts.
  - The retrieved context is used to prompt the LLM for SQL generation.

### LLM Integration
- **Google Gemini**: Used for keyword extraction, SQL generation, and conversational response. Prompts are carefully designed to ensure output format consistency and SQL Server compatibility.

### Data Flow
1. **Schema Extraction**: SQL Server schema is queried and embedded.
2. **Storage**: Embeddings are stored in Qdrant collections with rich metadata.
3. **Query Handling**: User question → keyword extraction → embedding → semantic search in Qdrant.
4. **Context Assembly**: Relevant schema and example SQLs are retrieved and assembled into a prompt for the LLM.
5. **SQL Generation & Execution**: LLM generates SQL, which is executed against the database.
6. **Conversational Response**: LLM generates a friendly answer using the query results and thought process.
