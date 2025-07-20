import os
import logging
import sys
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Define PROJECT_ROOT

os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "structured_content"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "indexing.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("indexer")


# Initialize Hugging Face Tokenizer and Model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333) # Connect to local Qdrant instance

def encode(texts):
    """Generates embeddings for a list of texts using Hugging Face Transformers."""
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Take the mean of the last hidden state for sentence embeddings
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

def load_documents(data_dir="data"):
    """Loads all text files from the specified directory."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": filepath})
    print(f"Loaded {len(documents)} documents from {data_dir}/")
    return documents

def chunk_documents(documents):
    """Chunks documents into smaller, manageable pieces."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        # add_start_index=True # We will calculate this manually
    )
    chunks_with_metadata = []
    for doc_item in documents:
        content = doc_item["content"]
        source = doc_item["source"]
        
        current_index = 0
        # Manually split text and calculate start index
        for chunk_text in text_splitter.split_text(content):
            start_index = content.find(chunk_text, current_index)
            if start_index == -1:
                # Should not happen if split_text works as expected, but a fallback
                start_index = current_index
            
            metadata = {
                "source": source,
                "starting_text_index": start_index
            }
            chunks_with_metadata.append({"text": chunk_text, "metadata": metadata})
            current_index = start_index + len(chunk_text)
            
    print(f"Created {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

def create_and_store_embeddings(chunks_with_metadata, collection_name="gitlab_handbook"):
    """Ingest files into the vector database"""
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        logger.info(f"No existing collection to delete or error deleting: {collection_name}. Error: {e}")
    
    # Create collection with specified vector parameters
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=model.config.hidden_size, distance=models.Distance.COSINE),
    )

    try:
        # Generate embeddings for chunks
        texts_to_encode = [item["text"] for item in chunks_with_metadata]
        embeddings = encode(texts_to_encode)
        
        # Prepare points for upsert operation
        points = [
            models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": item["text"],
                    "source": item["metadata"].get("source", "unknown"),
                    "start_index": item["metadata"].get("starting_text_index", -1) 
                }
            )
            for i, (embedding, item) in enumerate(zip(embeddings, chunks_with_metadata))
        ]
        
        # Upsert documents into Qdrant
        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        logger.info(f"Upsert operation info: {operation_info}")

    except Exception as e:
        logger.error(f"Error ingesting documents into {collection_name}: {str(e)}")
    
    return 

def list_and_describe_collections():
    """Lists all collections and provides a basic description."""
    print("\n--- Listing Qdrant Collections ---")
    try:
        response = qdrant_client.get_collections()
        collections = response.collections
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return

    if not collections:
        print("No collections found in Qdrant.")
        return

    print(f"Found {len(collections)} collection(s):")
    for collection_info in collections:
        try:
            collection_name = collection_info.name  # or collection_info['name'] depending on client version
            print(f"  - Collection Name: {collection_name}")

            # Fetch detailed collection info
            collection_details = qdrant_client.get_collection(collection_name)
            points_count = collection_details.points_count
            status = collection_details.status

            print(f"    Number of items: {points_count}")
            print(f"    Status: {status}")
        except Exception as e:
            print(f"    Error retrieving details for {collection_info}: {e}")


if __name__ == "__main__":
    loaded_docs = load_documents("sample_data")
    chunks = chunk_documents(loaded_docs)
    create_and_store_embeddings(chunks)
    print("Data processing pipeline initiated and embeddings stored in Qdrant.") 
    list_and_describe_collections()