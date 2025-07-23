# AI-Powered GitLab Handbook RAG System

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions based on the GitLab Handbook. It uses Qdrant in local mode for vector storage and Hugging Face Transformers for generating embeddings. The system is exposed via a Flask API, allowing for conversational interactions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Setup Details](#setup-details)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Environment Variables](#2-environment-variables)
  - [3. Local Qdrant Mode](#3-local-qdrant-mode)
- [Usage](#usage)
  - [1. Run the Application](#1-run-the-application)
  - [2. Interact with the API](#2-interact-with-the-api)
  - [3. Run the Chatbot Frontend](#3-run-the-chatbot-frontend)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)

## Features

- **Local Vector Database**: Uses Qdrant in local mode - no external server required!
- **Hugging Face Embeddings**: Generates document and query embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Document Chunking**: Utilizes `langchain`'s `RecursiveCharacterTextSplitter` to break down large documents into manageable chunks with metadata (source file path, start index).
- **Flask API**: Provides a RESTful API for interacting with the RAG system.
- **Conversation History**: Maintains chat history for contextual follow-up questions.
- **Gemini Integration**: Leverages the Gemini API (specifically `gemini-2.0-flash`) for generating refined responses based on retrieved context.
- **Logging**: Comprehensive logging for monitoring and debugging.
- **Instant Setup**: No Docker or external services required for development.
- **Chatbot Frontend**: A Streamlit-based interactive chat interface.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10+
- pip (Python package manager)

**Note**: No Docker or external Qdrant server required! The application uses Qdrant's local mode.

## Quick Start

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd assignment
   ```

2. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp example.env .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

That's it! The application will:
- Automatically create the local Qdrant database
- Load and index sample documents
- Start the Flask API server on http://localhost:8491

## Setup Details

### 1. Install Dependencies

Navigate to the backend directory and install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

The `requirements.txt` includes all necessary packages for local Qdrant mode:
- Core web framework (Flask)
- AI/ML models (Google Generative AI, OpenAI, transformers)
- Vector database (qdrant-client for local mode)
- Text processing (langchain)

### 2. Environment Variables

Copy the example environment file and configure it:

```bash
cp example.env .env
```

Edit the `.env` file and provide your Gemini API key:

```env
GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

Other variables are optional and have sensible defaults.

### 3. Local Qdrant Mode

The application uses Qdrant's local mode, which means:

- **No external server needed**: Qdrant runs embedded within the Python application
- **Persistent storage**: Data is stored in the `qdrant_data/` directory
- **Instant startup**: No waiting for Docker containers or network connections
- **Zero configuration**: Works out of the box

The local database is automatically created in `./qdrant_data/` when you first run the application.

## Usage

### 1. Run the Application

Navigate to the backend directory and start the Flask server:

```bash
cd backend
python app.py
```

The application will automatically:
- Create the local Qdrant database in `./qdrant_data/`
- Load and index documents from `sample_data/`
- Start the Flask API server on http://localhost:8491

### 2. Interact with the API

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
      "source": "sample_data/handbook_about_values.txt",
      "collection": "gitlab_handbook",
      "relevance_score": 0.85
    }
  ],
  "session_id": "test_session_123",
  "is_followup": false,
  "followup_confidence": 0.0,
  "history_length": 2
}
```

### 3. Run the Chatbot Frontend

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
      "top_k": 30,
      "session_id": "unique_session_id",
      "source": "your_client_type"
    }
    ```
  - **Response Body (JSON)**:
    ```json
    {
      "answer": "Generated response from the LLM.",
      "source_documents": [
        { 
          "content": "...", 
          "source": "file_path", 
          "collection": "collection_name", 
          "relevance_score": 0.8 
        }
      ],
      "session_id": "...",
      "is_followup": true/false,
      "followup_confidence": 0.0-1.0,
      "history_length": 2
    }
    ```

- **GET /health**
  - **Description**: Health check endpoint
  - **Response**: `{"status": "healthy"}`

## Technologies Used

- **Python 3.10+**: Core programming language
- **Flask**: Web framework for the API
- **Qdrant (Local Mode)**: Vector database for similarity search - no external server needed!
- **Hugging Face Transformers**: For generating text embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Google Generative AI (Gemini)**: LLM capabilities (`gemini-2.0-flash`)
- **LangChain**: For document splitting (`RecursiveCharacterTextSplitter`)
- **Streamlit**: For building the interactive chatbot frontend
- **python-dotenv**: For managing environment variables

## Project Structure

```
assignment/
├── backend/
│   ├── vector_database_handler.py  # Vector database operations
│   ├── app.py                      # Main Flask application  
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── app.py                      # Streamlit chat interface
│   └── requirements.txt            # Frontend dependencies
├── qdrant_data/                    # Local Qdrant database (auto-created)
├── .env                           # Environment configuration
├── example.env                    # Environment template
└── README.md                      # This file
```

## Benefits of Local Mode

- **🚀 Instant Setup**: No Docker or external services required
- **⚡ Better Performance**: Zero network latency, direct in-process calls
- **🔧 Simpler Deployment**: Single container/process
- **💡 Easy Development**: No complex infrastructure setup
- **📦 Portable**: Everything self-contained in one application

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Test locally using the simple setup
5. Submit a pull request

## License

This project is licensed under the MIT License.
