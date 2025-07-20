import os
import logging
import time
import threading
import uuid
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
import google.generativeai as genai

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
logger = logging.getLogger("model_config")

# Client cache to avoid duplicate initializations
_CLIENT_CACHE = {}

# Track completion calls in a single request flow
_ACTIVE_CALLS = {}
_ACTIVITY_LOCK = threading.Lock()
_SUPPRESSED_LOGS = False

def suppress_duplicate_logs():
    """Enable suppression of duplicate logs globally"""
    global _SUPPRESSED_LOGS
    _SUPPRESSED_LOGS = True
    logger.info("Duplicate log suppression enabled")

def enable_all_logs():
    """Disable suppression of duplicate logs globally"""
    global _SUPPRESSED_LOGS
    _SUPPRESSED_LOGS = False
    logger.info("Full logging enabled")

def start_request_flow(request_id=None):
    """Start tracking a new request flow, returns a request ID that can be used for 
    all related generate_completion calls."""
    if request_id is None:
        request_id = f"req_{uuid.uuid4()}"
    
    with _ACTIVITY_LOCK:
        _ACTIVE_CALLS[request_id] = {
            'started': time.time(),
            'is_primary': True,
            'children': [],
            'logged': False
        }
    
    logger.debug(f"Started new request flow: {request_id}")
    return request_id

def end_request_flow(request_id):
    """End tracking for a request flow and all its children"""
    if not request_id:
        return
    
    with _ACTIVITY_LOCK:
        if request_id in _ACTIVE_CALLS:
            # Clean up this request and all children
            for child_id in _ACTIVE_CALLS[request_id].get('children', []):
                if child_id in _ACTIVE_CALLS:
                    del _ACTIVE_CALLS[child_id]
            
            del _ACTIVE_CALLS[request_id]
            logger.debug(f"Ended request flow: {request_id}")

# Configure Generative AI (Gemini)
load_dotenv() # Ensure .env is loaded for API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEY not set.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
# Using 'gemini-2.0-flash' for text generation
GEMINI_MODEL = genai.GenerativeModel('gemini-2.0-flash')

def generate_completion(prompt: Union[str, list], system_message: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None,
                        parent_request_id: Optional[str] = None) -> str:
    """Generates a completion using the Gemini API."""
    logger.info(f"[ModelConfig] Generating completion for request ID: {request_id}")

    # Default generation configuration
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 4000,
        "top_p": 1,
        "top_k": 1
    }
    
    if config:
        # Map 'max_tokens' to 'max_output_tokens' if present
        if "max_tokens" in config:
            generation_config["max_output_tokens"] = config["max_tokens"]
            del config["max_tokens"] # Remove to avoid passing it directly

        generation_config.update(config) # Override defaults with remaining provided config
    
    # Prepare messages for the Gemini API
    messages = []
    if system_message:
        messages.append({"role": "user", "parts": [system_message]})
        # For Gemini, the system message often needs to be part of the initial user turn
        # or carefully crafted into the prompt if no explicit system role is supported for the model.
        # For 'gemini-pro', typically all instructions are in the 'user' turn.
        messages.append({"role": "model", "parts": ["Okay, I understand. How can I help?"]}) # A common practice for conversational models

    messages.append({"role": "user", "parts": [prompt]})
    
    try:
        response = GEMINI_MODEL.generate_content(
            contents=messages,
            generation_config=generation_config,
            # stream=True # Can enable streaming if needed
        )
        
        # Access the text from the response
        # Check if candidates exist and then get text from the first one
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            logger.warning(f"[ModelConfig] Gemini API returned no text in response for request ID: {request_id}")
            return "I apologize, but I could not generate a response at this time. Please try again."

    except Exception as e:
        logger.error(f"[ModelConfig] Error calling Gemini API for request ID {request_id}: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again later."

def clear_client_cache():
    """Clear the client cache for testing or memory management purposes."""
    global _CLIENT_CACHE
    _CLIENT_CACHE.clear()
    logger.info("Client cache cleared")

def clear_request_tracking():
    """Clear all request tracking data."""
    with _ACTIVITY_LOCK:
        global _ACTIVE_CALLS
        _ACTIVE_CALLS.clear()
    logger.info("Request tracking cleared")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        response = generate_completion(
            "Hello, what can you tell me about aelf blockchain?",
            system_message="You are a helpful assistant specializing in blockchain technology.",
            config={"temperature": 0.7, "max_tokens": 100}
        )
        print("Response:", response)
    except Exception as e:
        print(f"Test failed: {str(e)}") 