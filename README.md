# AI-Powered GitLab Handbook RAG System

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions based on the GitLab Handbook. It uses Qdrant as a vector database to store document embeddings and Hugging Face Transformers for generating embeddings. The system is exposed via a Flask API, allowing for conversational interactions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Qdrant Setup](#1-qdrant-setup)
  - [2. Project Dependencies](#2-project-dependencies)
  - [3. Environment Variables](#3-environment-variables)
- [Usage](#usage)
  - [1. Ingest Data into Vector Database](#1-ingest-data-into-vector-database)
  - [2. Run the API Server](#2-run-the-api-server)
  - [3. Interact with the API](#3-interact-with-the-api)
  - [4. Run the Chatbot Frontend](#4-run-the-chatbot-frontend)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)

## Features

- **Vector Database**: Uses Qdrant for efficient storage and retrieval of document embeddings.
- **Hugging Face Embeddings**: Generates document and query embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Document Chunking**: Utilizes `langchain`'s `RecursiveCharacterTextSplitter` to break down large documents into manageable chunks with metadata (source file path, start index).
- **Flask API**: Provides a RESTful API for interacting with the RAG system.
- **Conversation History**: Maintains chat history for contextual follow-up questions.
- **Gemini Integration**: Leverages the Gemini API (specifically `gemini-1.0-pro`) for generating refined responses based on retrieved context.
- **Logging**: Comprehensive logging for monitoring and debugging.
- **Chatbot Frontend**: A Streamlit-based interactive chat interface.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+
- Docker (for running Qdrant)

## Setup

### 1. Qdrant Setup

Run the Qdrant vector database using Docker. This will make Qdrant available on `localhost:6333`.

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify Qdrant is running by navigating to `http://localhost:6333/dashboard` in your browser.

### 2. Project Dependencies

Navigate to the project root directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, you can create it by running:

```bash
pip freeze > requirements.txt
```

And then install the necessary packages manually:

```bash
pip install python-dotenv langchain chromadb qdrant-client transformers torch google-generativeai flask
```

### 3. Environment Variables

Create a `.env` file in the root directory of your project (e.g., `assignment/.env`) and add your Gemini API key:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Replace `"YOUR_GEMINI_API_KEY"` with your actual API key obtained from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

### 1. Ingest Data into Vector Database

First, you need to load and embed your documents into the Qdrant database. The `sample_data` directory contains sample `.txt` files from the GitLab Handbook. You can replace these with your own data in a `data` directory.

```bash
python3 backend/vector_database_handler.py
```

This script will:
- Load text files from the `sample_data` directory.
- Chunk the documents.
- Generate embeddings using Hugging Face Transformers.
- Store the chunks and their metadata (source file, start index) in a Qdrant collection named `gitlab_handbook`.

### 2. Run the API Server

Start the Flask API server:

```bash
python3 backend/app.py
```

The server will typically run on `http://0.0.0.0:8491` (or the port specified in your `.env` file).

### 3. Interact with the API

Once the server is running, you can send queries to the chat endpoint. You can use tools like `curl`, Postman, or any HTTP client.

**Example Query (using curl):**

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"query": "Tell me about GitLab's values", "session_id": "test_session_123"}' \
http://localhost:8491/api/chat
```

**Example Response:**

```json
{
  "answer": "... (Gemini's generated response based on the context) ...",
  "source_documents": [
    {
      "content": "... (truncated content of a relevant document chunk) ...",
      "source": "data/handbook_about_values.txt",
      "collection": "gitlab_handbook",
      "relevance_score": 0.85
    }
    // ... more source documents
  ],
  "session_id": "test_session_123",
  "is_followup": false,
  "followup_confidence": 0.0,
  "history_length": 2
}
```

### 4. Run the Chatbot Frontend

Navigate to the `frontend` directory and install the required packages, then run the Streamlit application:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

This will launch the chatbot interface in your web browser, typically at `http://localhost:8501`.

## API Endpoints

- **POST /api/chat**
  - **Description**: Main chat endpoint for querying the RAG system.
  - **Request Body (JSON)**:
    ```json
    {
      "query": "Your question here",
      "max_files_per_repo": 30, // Optional: Number of top relevant documents to retrieve (default: 30)
      "session_id": "unique_session_id", // Optional: For maintaining conversation history
      "source": "your_client_type" // Optional: e.g., "lark_bot" for specific handling
    }
    ```
  - **Response Body (JSON)**:
    ```json
    {
      "answer": "Generated response from the LLM.",
      "source_documents": [
        { "content": "...", "source": "file_path", "collection": "collection_name", "relevance_score": 0.8 }
      ],
      "session_id": "...",
      "is_followup": true/false,
      "followup_confidence": 0.0-1.0,
      "history_length": integer
    }
    ```

## Technologies Used

- **Python**
- **Streamlit**: For building the interactive chatbot frontend.
- **streamlit-chat**: For creating chat UI components.
- **Requests**: For making HTTP requests to the backend API.
- **Flask**: Web framework for the API.
- **Qdrant**: Vector database for similarity search.
- **Hugging Face Transformers**: For generating text embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Google Generative AI (Gemini)**: For large language model capabilities (`gemini-1.0-pro`).
- **LangChain**: For document splitting (`RecursiveCharacterTextSplitter`).
- **python-dotenv**: For managing environment variables.
- **Docker**: For running Qdrant.
