import os
import sys # Import sys
import json
import logging
import time
import uuid
import torch
from tiktoken import encoding_for_model
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer
from qdrant_client_shared import get_qdrant_client
# Adjust project root for sys.path to resolve relative import issues when running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model configuration
from model_config import generate_completion, start_request_flow, end_request_flow, suppress_duplicate_logs # Changed to relative import for Docker
from vector_database_handler import load_docs_and_push_to_db
import re

# Determine the project root directory (one level up from services/) - adjusted for backend/app.py
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # This line is moved and modified above

# Ensure logs directory exists
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
# Create directory for local Qdrant database
os.makedirs(os.path.join(PROJECT_ROOT, "qdrant_data"), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app") # Changed from server to app

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Initialize Hugging Face Tokenizer and Model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode(texts):
    """Generates embeddings for a list of texts using Hugging Face Transformers."""
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Take the mean of the last hidden state for sentence embeddings
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store conversation history for each session
conversation_histories = {}

# Maximum conversation history length to maintain (increased from 10 to 5*2)
MAX_CONVERSATION_HISTORY = 10  # 5 exchanges = 10 messages (user+assistant)

# Conversation history cleanup interval (in seconds)
HISTORY_CLEANUP_INTERVAL = 86400  # 24 hours (was 3600 = 1 hour)

# Last cleanup timestamp
last_cleanup_time = time.time()

# Removed get_all_collections as it's no longer needed.

def parse_json_array_safely(response_text):
    """
    Safely parse a JSON array from the LLM response text, handling truncated or malformed JSON.
    Returns a list of strings or empty list if parsing fails.
    """
    response_text = response_text.strip()
    
    # Try direct JSON parsing first
    try:
        if response_text.startswith("[") and response_text.endswith("]"):
            return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Handle code block format
    if response_text.startswith("```") and "```" in response_text:
        # Extract content between code block markers
        parts = response_text.split("```")
        if len(parts) >= 3:
            code_content = parts[1]
            # Remove language identifier if present
            if code_content.startswith("json"):
                code_content = code_content[4:].strip()
            elif "\n" in code_content and code_content.split("\n", 1)[0].strip() in ["json", "JSON"]:
                code_content = code_content.split("\n", 1)[1].strip()
            
            try:
                return json.loads(code_content)
            except json.JSONDecodeError:
                pass
    
    # Try to find array content using regex
    import re
    array_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
    if array_match:
        try:
            array_content = array_match.group(0)
            return json.loads(array_content)
        except json.JSONDecodeError:
            pass
    
    # If all else fails, try to extract file paths using regex
    file_paths = []
    # Look for quoted strings that might be file paths
    path_matches = re.findall(r'"([^"]+\.[^"]+)"', response_text)
    if path_matches:
        for match in path_matches:
            if '/' in match and '.' in match:  # Basic check if it looks like a file path
                file_paths.append(match)
    
    return file_paths

def select_relevant_files(query, request_id=None):
    """
    Retrieves the most relevant documents directly from the Qdrant collection for the GitLab handbook.
    """
    logger.info(f"Retrieving relevant documents from Qdrant for query: {query}")
    
    try:
        collection_name = "gitlab_handbook"
        
        # Generate embedding for the query
        query_embedding = encode([query])[0] # encode returns a list of embeddings
        
        # Perform the similarity search using Qdrant client
        search_result = get_qdrant_client().search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=30, # Use the max_files parameter for the limit
            with_payload=True, # Retrieve payload (original text and metadata)
            # with_vectors=True # Not needed unless we want to inspect returned vectors
        )
        
        documents = []
        for hit in search_result:
            doc_content = hit.payload.get("text", "")
            # Qdrant stores metadata directly in the payload
            # Assuming 'path' is part of the payload if it was indexed
            file_path = hit.payload.get("source", f"chunk_{hit.id}.txt") # Default if path not in payload
            
            documents.append({
                'document': doc_content,
                'metadata': {'path': file_path},
                'collection': collection_name,
                'score': hit.score
            })
        
        logger.info(f"Retrieved {len(documents)} relevant documents from Qdrant.")
        return documents
        
    except Exception as e:
        print(f"Error retrieving documents from Qdrant: {e}")
        logger.error(f"Error retrieving documents from Qdrant: {e}")
        return []

def get_relevant_documents(query, request_id=None):

    try:
        all_documents = []
        
        # Directly call select_relevant_files with 'gitlab_handbook'
        gitlab_handbook_docs = select_relevant_files(
            query,
            request_id=request_id
        )
        
        all_documents.extend(gitlab_handbook_docs)
        logger.info(f"Added {len(gitlab_handbook_docs)} documents from gitlab_handbook")
        
        logger.info(f"Total documents selected: {len(all_documents)}")
        
        return all_documents
        
    except Exception as e:
        logger.error(f"Error getting relevant documents: {str(e)}")
        return []

def truncate_to_token_limit(text, max_tokens, encoding='gpt-4'):
    """
    Truncate the text to fit within the specified token limit.
    """
    tokenizer = encoding_for_model(encoding)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def summarize_text(text, max_tokens=200, parent_request_id=None):
    """
    Summarize the text to fit within a limited number of tokens.
    For very long texts, it first truncates before attempting summarization.
    """
    # Generate a request ID based on the input text (but tie it to the parent request)
    request_id = f"summarize_{hash(text[:100])}"
    
    # If text is very long, truncate it first to avoid context length errors
    # GPT-4o has a context length of ~128K tokens, but we use a much smaller limit for safety
    max_input_tokens = 12000
    truncated_text = truncate_to_token_limit(text, max_input_tokens)
    
    # If the text is still very large, use a simpler truncation
    if len(truncated_text) > 15000:  # ~3-4K tokens
        logger.info(f"Text too long for summarization ({len(truncated_text)} chars), using simple truncation")
        simple_summary = truncated_text[:5000] + "..."
        return simple_summary
    
    try:
        prompt = f"Summarize the following text to {max_tokens} tokens. Focus on extracting the key technical information, concepts, and code examples:\n\n{truncated_text}\n\nSummary:"
        
        system_message = "You are a technical assistant that summarizes documentation and code. Focus on preserving key technical details and code examples."
        
        return generate_completion(
            prompt,
            system_message=system_message,
            config={
                "max_tokens": max_tokens,
                "temperature": 0.3
            },
            request_id=request_id,
            parent_request_id=parent_request_id
        )
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        # Return a truncated version as fallback
        logger.info("Using fallback truncation for summarization")
        return truncated_text[:1000] + "..."

def generate_response_using_gemini(query, context_docs, max_context_tokens=8000, max_response_tokens=4000, request_id=None):
    """
    Generate a response using the configured model based on the query and retrieved context documents.
    """
    # Combine documents and their metadata
    logger.info(f"Starting to generate LLM response for query: {query[:50]}...")
    
    enhanced_docs = []
    for doc in context_docs:
        try:
            metadata = doc.get('metadata', {})
            file_path = metadata.get('path', 'unknown')
            
            # Skip README.md files (as a safety check)
            if file_path.lower().endswith('readme.md'):
                logger.debug(f"Skipping README file from context: {file_path}")
                continue
                
            collection = doc.get('collection', 'unknown')
            content = doc.get('document', '')
            
            # Add a more descriptive header for each document to make it clear to the LLM
            enhanced_doc = f"FILE: {collection}/{file_path}\n\n{content}"
            enhanced_docs.append({
                'content': enhanced_doc,
                'file_path': file_path,
                'collection': collection
            })
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
    
    logger.info(f"Processed {len(enhanced_docs)} documents for context")
    
    # Order documents by estimated relevance (currently all have equal weight)
    # In future could implement additional relevance scoring here
    
    docs_by_handbook = {}
    for doc in enhanced_docs:
        collection = doc['collection']
        if collection not in docs_by_handbook:
            docs_by_handbook[collection] = []
        docs_by_handbook[collection].append(doc)
    
    # Create more structured context
    structured_contexts = []
    for handbook, docs in docs_by_handbook.items():
        handbook_data_context = f"### {handbook.upper()} FILES:\n\n"
        for doc in docs:
            # For each document, include the full path and content
            content = doc['content']
            # Determine if we need to summarize based on length
            if len(content) > 1500:  # Only summarize long documents
                summarized = summarize_text(content, max_tokens=800, parent_request_id=request_id)
                handbook_data_context += summarized + "\n\n---\n\n"
            else:
                handbook_data_context += content + "\n\n---\n\n"
        structured_contexts.append(handbook_data_context)
    
    # Combine into a single string
    combined_context = "\n\n".join(structured_contexts)
    
    remaining_context_tokens = max_context_tokens
    
    # Truncate context to fit within token limit
    truncated_context = truncate_to_token_limit(combined_context, remaining_context_tokens)
    
    # Construct file list for reference
    file_list = "\n".join([f"- {doc['collection']}/{doc['file_path']}" for doc in enhanced_docs])
    
    full_context = f"SELECTED FILES FOR CONTEXT:\n{file_list}\n\nDETAILED FILE CONTENTS:\n{truncated_context}"
    
    # Final check to ensure we're within the limit
    context = truncate_to_token_limit(full_context, max_context_tokens)
    logger.info(f"Prepared context with {len(context)} characters")

    system_message = """You are an AI assistant specialized in the GitLab handbook ecosystem. \nYou provide accurate, helpful information based on the context provided.\nWhen answering:\n1. Review the list of selected files to understand what information is available\n2. Most importantly, use the DETAILED FILE CONTENTS to provide a comprehensive and detailed answer\n3. When answering coding questions, provide clear, working code examples based on the patterns in the file contents\n4. Cite the specific files you're referencing (e.g., "Based on src/Example.cs...")\n5. If you're unsure about something, acknowledge that but still try to provide helpful guidance based on what you do know\n6. DO NOT say "the context does not contain information" if relevant files have been selected - instead use those files to provide the best possible answer\n7. Format code blocks with proper syntax highlighting\n8. Use markdown for better readability\n9. Provide a complete, detailed, and helpful answer that would allow the user to implement the requested functionality"""

    prompt = f"""Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer the question based on the provided context. The files have been specifically selected as relevant to this question, so use their contents to provide a detailed and accurate response. \nInclude code examples that demonstrate how to implement the requested functionality using patterns from the selected files.\nDo not say that information is missing if files have been provided - use what's in the files to construct the best possible answer."""

    # Create final response request ID
    final_request_id = f"final_{request_id}" if request_id else None

    # Retry logic for API calls
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Starting backoff time in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Sending request to LLM API (attempt {retry_count + 1}/{max_retries})")
            
            # Set a timeout for the API call
            start_time = time.time()
            
            answer = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": 4000,
                    "temperature": 0.2,
                    # "timeout": 150  # 150 second timeout for the API call
                },
                request_id=final_request_id,
                parent_request_id=request_id
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"LLM API response received in {elapsed_time:.2f} seconds")
            
            logger.info(f"Generated response of {len(answer)} characters")
            return answer
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating LLM response (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                # If we've exhausted all retries, return a fallback response
                logger.warning("All retries exhausted. Returning fallback response.")
                return "I apologize, but I'm having trouble generating a response at the moment. The system might be experiencing high load or technical issues. Please try again in a few minutes."
            
            # Exponential backoff before retry
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

def cleanup_old_conversations():
    """
    Clean up conversation histories that haven't been used for a while.
    """
    global last_cleanup_time
    current_time = time.time()
    
    # Only run cleanup periodically
    if current_time - last_cleanup_time < HISTORY_CLEANUP_INTERVAL:
        return
        
    logger.info("Performing conversation history cleanup")
    
    # Sessions to remove
    to_remove = []
    
    for session_id, history_data in conversation_histories.items():
        last_updated = history_data.get('last_updated', 0)
        if current_time - last_updated > HISTORY_CLEANUP_INTERVAL:  # 24 hours of inactivity
            to_remove.append(session_id)
    
    # Remove old sessions
    for session_id in to_remove:
        conversation_histories.pop(session_id, None)
    
    logger.info(f"Cleaned up {len(to_remove)} inactive conversation histories")
    last_cleanup_time = current_time

def is_followup_question(query, conversation_history, request_id=None, retrieved_docs=None):
    """
    Use an LLM to determine if the current query is a follow-up question based on conversation history.
    If it is a follow-up, generate and return a response directly.
    
    Parameters:
    - query: The user's question
    - conversation_history: Past conversation messages
    - request_id: Tracking ID for the request
    - retrieved_docs: Previously retrieved documents (for follow-up responses)
    
    Returns a tuple:
    - is_followup: Boolean indicating if it's a follow-up question
    - confidence: Confidence score (0-1) of the follow-up classification
    - answer: Response string if it's a follow-up, None otherwise
    """
    # Perform quick check for obvious follow-up indicators before making an LLM call
    query_lower = query.lower()
    obvious_indicators = [
        "above step", "previous answer", "you provided", "from previous", 
        "the code you showed", "implement above", "previous step",
        "previous response", "earlier response", "last step"
    ]
    
    is_followup = False
    confidence = 0.0
    
    for indicator in obvious_indicators:
        if indicator in query_lower:
            logger.info(f"Detected obvious follow-up indicator: '{indicator}'")
            is_followup = True
            confidence = 0.95
            break
    
    # If there's no history or just one query, it can't be a follow-up
    if not conversation_history or len(conversation_history) < 3:
        logger.info("Not enough conversation history for follow-up detection")
        return False, 0.0, None
    
    # Log the conversation history being analyzed
    logger.info(f"Analyzing conversation history for follow-up detection ({len(conversation_history)} messages)")
    for i, entry in enumerate(conversation_history[-6:]):  # Log last 3 exchanges at most
        role = entry.get('role', 'unknown')
        content = entry.get('content', '')
        truncated_content = content[:100] + "..." if len(content) > 100 else content
        logger.info(f"  History[{i}] - {role}: {truncated_content}")
    
    # If not already detected as follow-up, use LLM to determine
    if not is_followup:
        # Create a simplified view of the conversation for the LLM
        conversation_summary = []
        for entry in conversation_history[:-1]:  # Exclude current query
            prefix = "User: " if entry['role'] == 'user' else "Assistant: "
            # Truncate very long entries to keep the prompt manageable
            content = entry['content']
            if len(content) > 500:
                content = content[:300] + "..." + content[-200:]
            conversation_summary.append(f"{prefix}{content}")
        
        conversation_text = "\n\n".join(conversation_summary)
        
        # Create a sub-request ID for tracking
        sub_request_id = f"followup_detection_{hash(query)}"
        
        prompt = f"""Given the following conversation history and a new user query, determine if the new query is a follow-up question that builds upon, refers to, or seeks clarification about information in the previous conversation.\n\nConversation History:\n{conversation_text}\n\nNew Query:\n{query}\n\nA follow-up question might:\n1. Explicitly refer to previous answers\n2. Ask for clarification or elaboration on something mentioned before\n3. Request implementation of a feature discussed previously\n4. Request modifications to code or solutions provided earlier\n5. Ask about specific details mentioned in previous responses\n6. Reference steps, numbers, or sections from a previous answer\n7. Ask how to implement specific parts from a prior response\n\nAnalyze the new query in relation to the conversation history and determine if it's a follow-up question.\nOutput your decision as "true" or "false" followed by a confidence score between 0 and 1.\nExample: "true,0.95" or "false,0.75"\n"""

        system_message = """You are an AI that analyzes conversations to determine if a new query is a follow-up question to previous exchanges. Your task is to identify when a user is referencing, building upon, or asking for more details about something mentioned in earlier messages. Be especially attentive to cases where the user is asking about specific steps, sections, or code mentioned in previous responses. Output only your decision as true/false with a confidence score."""

        try:
            response = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                request_id=sub_request_id,
                parent_request_id=request_id
            )
            
            # Parse the response to get the decision and confidence
            response = response.strip().lower()
            logger.info(f"LLM follow-up detection response: {response}")
            
            if "true" in response:
                # Extract confidence score if present
                confidence = 0.9  # Default high confidence
                try:
                    # Try to extract confidence from patterns like "true,0.85"
                    if "," in response:
                        confidence_str = response.split(",")[1].strip()
                        confidence = float(confidence_str)
                except:
                    logger.warning("Could not parse confidence score from LLM response")
                
                is_followup = True
            else:
                # Extract confidence score if present
                confidence = 0.9  # Default high confidence
                try:
                    # Try to extract confidence from patterns like "false,0.85"
                    if "," in response:
                        confidence_str = response.split(",")[1].strip()
                        confidence = float(confidence_str)
                except:
                    logger.warning("Could not parse confidence score from LLM response")
                
                is_followup = False
                
        except Exception as e:
            logger.error(f"Error in LLM follow-up detection: {str(e)}")
            
            # Fall back to a smarter heuristic detection in case of API error
            query_lower = query.lower()
            followup_indicators = [
                "previous", "above", "earlier", "you said", "your answer",
                "step", "implement", "provided", "mentioned", "showed",
                "code", "example", "modify", "change", "update", "fix"
            ]
            
            # Count how many indicators are present
            indicator_count = sum(1 for indicator in followup_indicators if indicator in query_lower)
            
            # If multiple indicators are found, very likely a follow-up
            if indicator_count >= 2:
                logger.info(f"Fallback detected {indicator_count} follow-up indicators in query")
                is_followup = True
                confidence = 0.8
            
            # If even one indicator is found with a recent conversation, probably a follow-up
            elif indicator_count >= 1 and len(conversation_history) >= 3:
                logger.info(f"Fallback detected 1 follow-up indicator with recent conversation")
                is_followup = True
                confidence = 0.7
                
            # If the conversation has multiple exchanges within the last 5 minutes, lean toward classifying as follow-up
            elif len(conversation_history) >= 5:
                logger.info("Fallback classified as follow-up based on active conversation")
                is_followup = True
                confidence = 0.6
            else:
                is_followup = False
                confidence = 0.5  # Low confidence when falling back to heuristics
    
    # If this is a follow-up question and we have retrieved documents, generate a response
    if is_followup and retrieved_docs:
        logger.info("Generating response for follow-up question")
        answer = generate_followup_response(query, retrieved_docs, conversation_history, request_id)
        return is_followup, confidence, answer
    
    # If it's a follow-up but we don't have retrieved_docs, just return the classification
    return is_followup, confidence, None

def generate_followup_response(query, context_docs, conversation_history, request_id=None):
    """
    Generate a response for a follow-up question, focusing on conversation history.
    This function is separate from the main response generation flow to keep concerns separated.
    """
    logger.info("Using dedicated follow-up question processing")
    
    # Log conversation history being used in generation
    if conversation_history:
        logger.info(f"Using conversation history with {len(conversation_history)} messages for follow-up response")
        for i, entry in enumerate(conversation_history[-6:]):  # Log last 3 exchanges (6 messages)
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            content_preview = content[:100] + "..." if len(content) > 100 else content
            logger.info(f"  ConvHistory[{i}] - {role}: {content_preview}")
    
    # Process context documents
    enhanced_docs = []
    for doc in context_docs:
        try:
            metadata = doc.get('metadata', {})
            file_path = metadata.get('path', 'unknown')
            
            # Skip README.md files (as a safety check)
            if file_path.lower().endswith('readme.md'):
                logger.debug(f"Skipping README file from context: {file_path}")
                continue
                
            collection = doc.get('collection', 'unknown')
            content = doc.get('document', '')
            
            # Add a more descriptive header for each document to make it clear to the LLM
            enhanced_doc = f"FILE: {collection}/{file_path}\n\n{content}"
            enhanced_docs.append({
                'content': enhanced_doc,
                'file_path': file_path,
                'collection': collection
            })
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
    
    logger.info(f"Processed {len(enhanced_docs)} documents for follow-up context")
    
    # Create file list for reference
    file_list = "\n".join([f"- {doc['collection']}/{doc['file_path']}" for doc in enhanced_docs])
    
    # Prepare the conversation history context with higher prominence
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        # For follow-ups, include more detailed conversation history
        history_entries = []
        for entry in conversation_history:
            if entry['role'] == 'user':
                history_entries.append(f"USER: {entry['content']}")
            else:  # 'assistant'
                history_entries.append(f"ASSISTANT: {entry['content']}")
        
        conversation_context = "CONVERSATION HISTORY:\n" + "\n\n".join(history_entries) + "\n\n"
        logger.info(f"Including detailed conversation history with {len(conversation_history)} exchanges for follow-up")
    
    # Build a focused context for follow-up questions that prioritizes conversation history
    context = f"""CONVERSATION HISTORY:\n{conversation_context}\n\nAVAILABLE FILES FOR REFERENCE:\n{file_list}\n\nCODE CONTEXT SUMMARY:\n This has all data of gitlab handbook. The complete file list is provided above, and specific code details\ncan be referenced when needed."""

    # Use a system message focused on iterative improvement
    system_message = """You are an AI assistant specialized in the GitLab handbook ecosystem.\nYou are responding to a follow-up question where the user wants you to improve, modify, or build upon your previous response.\n\nWhen answering follow-up questions:\n1. Carefully analyze the conversation history to understand what the user is asking you to improve or implement\n2. If they're asking you to implement TODOs or suggestions you made previously, use your earlier guidance to provide concrete implementations\n3. If they're asking for improvements to code you provided, revise the code with the requested enhancements\n4. Be precise and direct in addressing what the user is asking for\n5. Prefer to build upon your previous answers rather than starting from scratch\n6. If you're improving code, provide complete implementations, not just snippets or outlines\n7. Ensure your response is immediately useful and actionable"""

    # Special prompt for follow-up questions
    prompt = f"""CONVERSATION HISTORY AND CONTEXT:\n{context}\n\nCURRENT QUERY:\n{query}\n\nPlease respond to this follow-up question by directly addressing what the user is asking for.\nIf they want you to implement TODOs or suggestions you mentioned earlier, provide a complete implementation.\nIf they want you to improve code you provided earlier, make those improvements and explain what changed.\nYour response should be specific, actionable, and build upon the conversation history."""

    # Create final response request ID
    final_request_id = f"followup_{request_id}" if request_id else None

    # Retry logic for API calls
    max_retries = 3
    retry_count = 0
    backoff_time = 2  # Starting backoff time in seconds
    
    while retry_count < max_retries:
        try:
            logger.info(f"Sending follow-up request to LLM API (attempt {retry_count + 1}/{max_retries})")
            
            # Set a timeout for the API call
            start_time = time.time()
            
            answer = generate_completion(
                prompt,
                system_message=system_message,
                config={
                    "max_tokens": 4000,
                    "temperature": 0.2,
                    # "timeout": 150  # 150 second timeout for the API call
                },
                request_id=final_request_id,
                parent_request_id=request_id
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Follow-up LLM API response received in {elapsed_time:.2f} seconds")
            
            logger.info(f"Generated follow-up response of {len(answer)} characters")
            return answer
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating follow-up LLM response (attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                # If we've exhausted all retries, return a fallback response
                logger.warning("All retries exhausted. Returning fallback response for follow-up.")
                return "I apologize, but I'm having trouble generating a follow-up response at the moment. The system might be experiencing high load or technical issues. Please try again in a few minutes."
            
            # Exponential backoff before retry
            logger.info(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for Docker and monitoring systems.
    """
    try:
        # Basic check if the app is responsive
        return jsonify({"status": "healthy", "message": "Service is running"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """
    API endpoint for chat functionality.
    Expects JSON with format: {"query": "your question here", "top_k": 20, "session_id": "unique_session_id"}
    Now supports conversation history for iterative improvements.
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' parameter"}), 400
        
        query = data['query']
        
        # Get session ID for conversation tracking (defaults to a new random ID if not provided)
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        logger.info(f"Received query in session {session_id}: {query}")
        
        # Initialize or get conversation history for this session
        if session_id not in conversation_histories:
            logger.info(f"Creating new conversation history for session {session_id}")
            conversation_histories[session_id] = {
                'history': [],
                'last_updated': time.time(),
                'last_documents': [],  # Store the last retrieved documents
                'last_timestamp': time.time()
            }
        else:
            logger.info(f"Using existing conversation history for session {session_id} with {len(conversation_histories[session_id]['history'])} messages")
        
        # Update the last access time
        conversation_histories[session_id]['last_updated'] = time.time()
        
        # Get the conversation history
        conversation_history = conversation_histories[session_id]['history']
        
        # Add the user query to history
        conversation_history.append({
            'role': 'user',
            'content': query
        })
        
        # Log conversation state after adding the new query
        logger.info(f"Conversation history now has {len(conversation_history)} messages")
        
        # Trim history if it gets too long
        if len(conversation_history) > MAX_CONVERSATION_HISTORY * 2:  # *2 because each exchange has user and assistant
            logger.info(f"Trimming conversation history from {len(conversation_history)} to {MAX_CONVERSATION_HISTORY*2} messages")
            conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY*2:]
            conversation_histories[session_id]['history'] = conversation_history
        
        # Start a request flow for this chat query
        request_id = start_request_flow(f"chat_{session_id}_{hash(query)}")
        
        # Step 1: First check if we have documents from a previous query for this session
        # This will be used for follow-up detection and handling
        previous_docs = conversation_histories[session_id].get('last_documents', [])
        
        # Step 2: Determine if this is a follow-up question using pattern detection first, then LLM
        is_followup = False
        followup_confidence = 0.0
        followup_answer = None
        
        # Check for explicit follow-up flag in the request
        if data.get('is_followup', False):
            is_followup = True
            followup_confidence = 1.0
            logger.info("Follow-up explicitly indicated in API request")
            
            # If we have documents, generate follow-up response
            if previous_docs:
                followup_answer = generate_followup_response(query, previous_docs, conversation_history, request_id)
        else:
            # Use specialized follow-up handler to determine if it's a follow-up and generate a response if needed
            is_followup, followup_confidence, followup_answer = is_followup_question(
                query, 
                conversation_history, 
                request_id=request_id,
                retrieved_docs=previous_docs
            )
            logger.info(f"Follow-up detection: {is_followup} (confidence: {followup_confidence})")
        
        # Update timestamp for this interaction
        conversation_histories[session_id]['last_timestamp'] = time.time()
        
        # If we recognized this as a follow-up and already generated a response, use it
        if is_followup and followup_answer:
            logger.info("Using pre-generated follow-up response")
            answer = followup_answer
            
            # Add the assistant's response to history
            conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            # Format source documents for the response (using previous documents)
            source_documents = []
            for doc in previous_docs:
                source_documents.append({
                    "content": doc.get('document', '')[:200] + "...",  # Truncated preview
                    "source": doc.get('metadata', {}).get('path', 'unknown'),
                    "collection": doc.get('collection', 'unknown'),
                    "relevance_score": 1.0  # All LLM-selected files have equal base relevance
                })
            
            # Clean up old conversation histories periodically
            cleanup_old_conversations()
            
            logger.info(f"Returning follow-up response with {len(source_documents)} source documents, history length: {len(conversation_history)}")
            
            return jsonify({
                "answer": answer,
                "source_documents": source_documents,
                "session_id": session_id,
                "is_followup": is_followup,
                "followup_confidence": followup_confidence,
                "history_length": len(conversation_history)
            })
        
        # If not a follow-up or couldn't generate a follow-up response, proceed with normal flow
        try:
            # Retrieve fresh documents for a new query
            retrieved_docs = get_relevant_documents(
                query, 
                request_id=request_id
            )
            
            # Store these documents for potential future follow-up questions
            conversation_histories[session_id]['last_documents'] = retrieved_docs
            
            if not retrieved_docs:
                answer = "I couldn't find any relevant information in the gitlab handbook."
                
                # Add the assistant's response to history
                conversation_history.append({
                    'role': 'assistant',
                    'content': answer
                })
                
                end_request_flow(request_id)  # Clean up tracking
                return jsonify({
                    "answer": answer,
                    "source_documents": [],
                    "session_id": session_id,
                    "is_followup": is_followup,
                    "history_length": len(conversation_history)
                })
            
            # Generate response using GPT-4 for a new query (no conversation history needed)
            answer = generate_response_using_gemini(
                query, 
                retrieved_docs, 
                request_id=request_id
            )
            
            # Log the response before adding to history
            logger.info(f"Response generated, adding to conversation history. History will have {len(conversation_history) + 1} messages")
            
            # Add the assistant's response to history
            conversation_history.append({
                'role': 'assistant',
                'content': answer
            })
            
            # Format source documents for the response
            source_documents = []
            for doc in retrieved_docs:
                source_documents.append({
                    "content": doc.get('document', '')[:200] + "...",  # Truncated preview
                    "source": doc.get('metadata', {}).get('path', 'unknown'),
                    "collection": doc.get('collection', 'unknown'),
                    "relevance_score": 1.0  # All LLM-selected files have equal base relevance
                })
            
            # Log the full response for debugging
            logger.info(f"Returning answer with {len(source_documents)} source documents (is_followup: {is_followup}, history_length: {len(conversation_history)})")
            
            # Clean up old conversation histories periodically
            cleanup_old_conversations()
            
            return jsonify({
                "answer": answer,
                "source_documents": source_documents,
                "session_id": session_id,
                "is_followup": is_followup,
                "followup_confidence": followup_confidence,
                "history_length": len(conversation_history)
            })
        finally:
            # End the request flow to clean up tracking
            end_request_flow(request_id)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def list_and_describe_collections():
    """Lists all collections and provides a basic description."""
    print("\n--- Listing Qdrant Collections ---")
    try:
        response = get_qdrant_client().get_collections()
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
            collection_name = collection_info.name
            print(f"  - Collection Name: {collection_name}")

            # Fetch detailed collection info to get the points count
            collection_details = get_qdrant_client().get_collection(collection_name)
            points_count = collection_details.points_count
            status = collection_details.status

            print(f"    Number of items: {points_count}")
            print(f"    Status: {status}")
        except Exception as e:
            print(f"    Error retrieving details for {collection_info.name}: {e}")


def main():
    # list_and_describe_collections()
    load_docs_and_push_to_db()

    # Load configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8491))
    
    # Enable duplicate log suppression
    suppress_duplicate_logs()
    
    # Start the Flask app
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main() 