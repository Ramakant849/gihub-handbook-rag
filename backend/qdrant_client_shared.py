"""
Shared Qdrant client module to avoid conflicts between different parts of the application.
"""
import os
import logging
import time
from qdrant_client import QdrantClient

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logger = logging.getLogger("qdrant_shared")

# Global client instance
_qdrant_client = None

def create_qdrant_client():
    """Create Qdrant client using local mode (no external server needed)"""
    # Option 1: Use persistent local storage
    local_db_path = os.path.join(PROJECT_ROOT, "qdrant_data")
    
    # Option 2: Use in-memory storage (uncomment the line below and comment the line above)
    # local_db_path = ":memory:"
    
    logger.info(f"Creating local Qdrant client with path: {local_db_path}")
    
    try:
        client = QdrantClient(path=local_db_path)
        # Test the client by getting collections (this will work even if no collections exist)
        collections = client.get_collections()
        logger.info(f"Successfully created local Qdrant client. Found {len(collections.collections)} existing collections.")
        return client
    except Exception as e:
        logger.error(f"Failed to create local Qdrant client: {e}")
        raise

def get_qdrant_client():
    """Get or create the shared Qdrant client (singleton pattern)"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = create_qdrant_client()
    return _qdrant_client

def reset_client():
    """Reset the client (useful for testing)"""
    global _qdrant_client
    _qdrant_client = None 