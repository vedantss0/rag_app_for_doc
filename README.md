# Project Proposal Assistant

A Retrieval-Augmented Generation (RAG) pipeline that helps users draft project proposals from project documents. The application uses LangChain and LangGraph to extract key details (fine-prints) from documents and provides a chatbot interface to answer user queries.

## Features

- Document processing for PDF files
- Fine-prints extraction to identify key details in project documents
- Conversational interface for users to ask questions about the documents
- FastAPI endpoints for integrating with other applications

## Technical Architecture

- **Frameworks**: LangChain + LangGraph
- **API**: FastAPI
- **LLM Options**: Google Gemini (default) or OpenAI
- **Vector Store**: FAISS
- **Embeddings**: OpenAI embeddings or local HuggingFace embeddings

## LangGraph Pipeline Flow

The application uses LangGraph to create explicit, structured workflows:

1. **Fine-Prints Extraction**:
   - `START` → `process_documents` → `extract_fine_prints`

2. **Chat Interface**:
   - `START` → `retrieve` → `generate`

This approach makes the RAG pipeline more modular, testable, and maintainable.

## Prerequisites

- Python 3.8+
- PDF documents in the `data` folder

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-proposal-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to add your API keys:
   ```
   LLM_TYPE=gemini  # gemini, openai, or other
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Place your PDF documents in the `data` folder

## Running the Application

Start the FastAPI server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`.

## API Endpoints

### GET /fine-prints

Returns extracted fine-prints from all project documents.

Example:
```bash
curl -X GET http://localhost:8000/fine-prints
```

### POST /chat

Accepts user queries and returns responses from the chatbot.

Example:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key deliverables for this project?"}'
```

## Output Files

- `fine_prints.txt`: Contains all extracted fine-prints from the documents
- `chat_response.txt`: Contains a log of all chat queries and responses

## Customization

- Change LLM: Edit the `.env` file to switch between Gemini and OpenAI
- Adjust document processing: Modify chunk size and overlap in `config.py`
- Custom prompts: Edit the prompt templates in the graph classes
- Extend the graph: Add new nodes to the LangGraph workflows for additional functionality

## Troubleshooting

- **No PDF files found**: Ensure you have placed PDF documents in the `data` folder
- **API key errors**: Verify your API keys in the `.env` file
- **Memory issues**: For large documents, reduce the chunk size in `config.py`