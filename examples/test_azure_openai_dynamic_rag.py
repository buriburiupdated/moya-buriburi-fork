"""
Interactive Azure OpenAI chat interface with dynamic RAG and web search capabilities.
Combines features from test_azure_openai.py and quick_start_dynamic_rag.py. almost there
"""

import os
import sys
import json
import tempfile
import time
import re
import dotenv
from openai import AzureOpenAI
from langchain_ollama import OllamaEmbeddings

from moya.tools.tool_registry import ToolRegistry
from moya.tools.rag_search_tool import VectorSearchTool
from moya.tools.search_tool import SearchTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.conversation.message import Message
from moya.conversation.thread import Thread
from moya.vectorstore.faisscpu_vectorstore import FAISSCPUVectorstoreRepository

dotenv.load_dotenv()

def clean_temp_files():
    """Clean up any temporary files."""
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith('.txt') and file.startswith('tmp'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass

def setup_vector_store(store_path="azure-dynamic-faiss"):
    try:
        # Create embeddings using Ollama
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create path if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize vector store
        vector_store = FAISSCPUVectorstoreRepository(path=store_path, embeddings=embeddings)
        
        # Create the vector store if it doesn't exist
        if not os.path.exists(os.path.join(store_path, "index.faiss")):
            print("Creating new vector store...")
            vector_store.create_vectorstore()
            
            # Test if vector store is working - FIXED VERSION
            from langchain_core.documents import Document
            test_text = "This is a test document for the vector store."
            test_doc = Document(page_content=test_text, metadata={"source": "test"})
            vector_store.add_vector([test_doc])
            print("Vector store initialized and tested successfully")
        else:
            print("Loading existing vector store...")
            
        return vector_store
    except Exception as e:
        print(f"\nError setting up vector store: {str(e)}")
        print("Make sure Ollama is running and nomic-embed-text model is available")
        print("Try: ollama pull nomic-embed-text")
        sys.exit(1)

def add_text_to_knowledge_base(vector_store, text, source="user_input"):
    """Add new text to the knowledge base."""
    try:
        # Create a temporary file to store the text (approach from quick_start_dynamic_rag.py)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(text)
            temp_path = temp_file.name
        
        # Load the temporary file into the vector store
        vector_store.load_file(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        return True
    except Exception as e:
        print(f"Error adding text to knowledge base: {e}")
        return False

def create_azure_client():
    """Create and return an Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    print("\nChecking Azure OpenAI configuration:")
    print(f"API Base: {api_base}")
    print(f"API Version: {api_version}")
    print(f"API Key: {'Set' if api_key else 'Not Set'}")
    
    # Verify credentials are set
    if not api_key or not api_base:
        print("Error: Azure OpenAI credentials not set. Please set environment variables:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION (optional)")
        sys.exit(1)
        
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )

def extract_and_add_search_info(client, vector_store, query, search_result):
    """Extract valuable information from search results and add to knowledge base."""
    try:
        # Use Azure OpenAI to extract key information
        messages = [
            {"role": "system", "content": "Extract key factual information from the search results."},
            {"role": "user", "content": f"Extract key facts from this search result about '{query}': {search_result[:2000]}..."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Use your actual Azure deployment name
            messages=messages,
            temperature=0.3
        )
        
        extracted_info = response.choices[0].message.content
        formatted_info = f"# Information about: {query}\n\n{extracted_info}\n\nSource: Web search"
        
        # Add to knowledge base
        success = add_text_to_knowledge_base(vector_store, formatted_info, source="web_search")
        
        if success:
            print(f"[LEARNING] Added new information about '{query}' to knowledge base")
            return extracted_info
        else:
            print(f"[ERROR] Failed to add information to knowledge base")
            return None
            
    except Exception as e:
        print(f"Error extracting information: {e}")
        return None

def perform_search(query):
    """Directly perform a web search and return the results."""
    print(f"\n[SEARCH] Searching for: {query}")
    result = SearchTool.search_web_free(query)
    print(f"[SEARCH] Search completed")
    return result

def format_conversation_context(messages):
    """Format conversation history for context."""
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context

def main():
    """Run the interactive Azure OpenAI chat with dynamic RAG."""
    print("Initializing Azure OpenAI with Dynamic RAG...")
    
    # Test search functionality
    print("Testing search functionality...")
    try:
        test_result = SearchTool.search_web_free("test query azure openai")
        print("Direct test search completed")
    except Exception as e:
        print(f"Warning: Search test failed: {e}")
    
    # Initialize Azure OpenAI client
    client = create_azure_client()
    
    # Set up vector store for RAG
    vector_store = setup_vector_store()
    
    # Add initial information to knowledge base
    initial_info = """
    # About Azure OpenAI Dynamic RAG
    
    This system combines Azure OpenAI with dynamic Retrieval-Augmented Generation (RAG).
    
    Key features:
    1. Knowledge Base - Stores and retrieves information using vector embeddings
    2. Web Search - Falls back to searching the web when information isn't found locally
    3. Dynamic Learning - Adds new information to the knowledge base during conversations
    4. Special Commands - Supports commands for direct interaction with the system
    
    This implementation uses Azure OpenAI for text generation and Ollama for embeddings.
    """
    add_text_to_knowledge_base(vector_store, initial_info, source="initial_setup")
    
    # Set up conversation thread
    thread_id = "azure_rag_chat"
    EphemeralMemory.memory_repository.create_thread(Thread(thread_id=thread_id))
    
    # Initialize messages array with system prompt
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with knowledge base and web search capabilities. Use <search query=\"your query\"> to search your knowledge base and <web_search query=\"your query\"> to search the web when needed."}
    ]
    
    # Store system message
    EphemeralMemory.store_message(
        thread_id=thread_id,
        sender="system",
        content="Conversation initialized. Knowledge base and search capability enabled."
    )
    
    # Display welcome message
    print("\n" + "=" * 80)
    print("Welcome to Azure OpenAI Chat with Dynamic RAG!")
    print("This assistant can:")
    print("- Search its knowledge base for information")
    print("- Search the web when information isn't in the knowledge base")
    print("- Learn new information that you provide")
    print("- Automatically add information from web searches to its knowledge base")
    print("\nSpecial commands:")
    print("- /add [information] - Add new information to the knowledge base")
    print("- /search [query] - Directly search the web")
    print("- 'exit' or 'quit' - End the conversation")
    print("=" * 80)
    
    # Main conversation loop
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
            
        # Handle add command
        if user_input.startswith("/add "):
            new_info = user_input[5:].strip()
            if new_info:
                success = add_text_to_knowledge_base(vector_store, new_info)
                
                # Store user message and system response
                EphemeralMemory.store_message(
                    thread_id=thread_id,
                    sender="user",
                    content=f"Please add this information to your knowledge base: {new_info}"
                )
                
                response = "Information successfully added to my knowledge base. I can now use this in future responses."
                EphemeralMemory.store_message(
                    thread_id=thread_id,
                    sender="assistant",
                    content=response
                )
                
                print(f"\nAssistant: {response}")
                continue
                
        # Handle direct search command
        if user_input.startswith("/search "):
            search_query = user_input[8:].strip()
            search_result = perform_search(search_query)
            
            # Auto-learn from search result
            print("\n[LEARNING] Extracting and adding information to knowledge base...")
            extracted_info = extract_and_add_search_info(client, vector_store, search_query, search_result)
            
            print("\nDirect Search Result:")
            print(search_result[:500] + "..." if len(search_result) > 500 else search_result)
            print("\nInformation has been added to the knowledge base.")
            continue
        
        # Add this to your main() function to implement a /kb command
        if user_input.startswith("/kb "):
            search_query = user_input[4:].strip()
            print(f"\n[Directly searching knowledge base for: '{search_query}']")
            
            # Use langchain FAISS directly to bypass problematic code
            from langchain_community.vectorstores import FAISS
            from langchain_ollama import OllamaEmbeddings
            
            # Load knowledge base directly
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            try:
                # Add the allow_dangerous_deserialization parameter
                direct_vs = FAISS.load_local(
                    "azure-dynamic-faiss", 
                    embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter
                )
                results = direct_vs.similarity_search(search_query, k=3)
                
                # Display results
                print("\nKnowledge Base Results:")
                for i, doc in enumerate(results):
                    print(f"\nDocument {i+1}:")
                    print(f"{doc.page_content}")
                
            except Exception as e:
                print(f"Error accessing knowledge base: {e}")
        
        # For any factual question, automatically trigger a search
        if any(keyword in user_input.lower() for keyword in ["who", "what", "when", "where", "how", "why"]):
            query = user_input
            print(f"\n[Auto-searching for: '{query}']")
            vector_search_matches = [query]  # Force vector search

        # Store user message
        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="user",
            content=user_input
        )
        
        # Retrieve conversation context
        thread = EphemeralMemory.memory_repository.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)
        conversation_context = format_conversation_context(previous_messages)
        
        # Enhanced input with instruction to use RAG search and web search
        enhanced_input = (
            f"{conversation_context}"
            f"User: {user_input}\n\n"
            f"Remember to first search the knowledge base and then the web if needed."
        )
        
        # Add user message to messages array
        messages.append({"role": "user", "content": enhanced_input})
        
        print("\nAssistant: ", end="", flush=True)
        start_time = time.time()
        
        try:
            # Get initial response
            response = client.chat.completions.create(
                model="gpt-4o",  # Use your actual Azure deployment name
                messages=messages
            )
            
            full_response = response.choices[0].message.content
            
            # Check for search patterns
            vector_search_pattern = r'<search query="([^"]+)">'
            web_search_pattern = r'<web_search query="([^"]+)">'
            
            vector_search_matches = re.findall(vector_search_pattern, full_response)
            web_search_matches = re.findall(web_search_pattern, full_response)
            
            print(f"DEBUG - Vector search matches: {vector_search_matches}")
            print(f"DEBUG - Web search matches: {web_search_matches}")
            
            # If searches were requested, process them and regenerate response
            if vector_search_matches or web_search_matches:
                # Clear the previous output
                if os.name == 'posix':  # For Linux/Mac
                    print("\033[F\033[K" * (full_response.count('\n') + 1), end="")
                else:  # For Windows
                    os.system('cls' if os.name == 'nt' else 'clear')

                print(f"\nYou: {user_input}")
                
                search_context = ""
                learned_info = []  # Track what we've learned
                
                # Process vector searches
                for query in vector_search_matches:
                    print(f"\n[Searching knowledge base for: '{query}']")
                    
                    # Execute vector search
                    tool_registry = ToolRegistry()
                    VectorSearchTool.configure_vector_search_tools(tool_registry)
                    tool = tool_registry.get_tool("VectorSearchTool")
                    search_results_json = tool.function(
                        query=query,
                        vector_store=vector_store,  # Pass the vector_store object directly
                        k=5
                    )
                    search_data = json.loads(search_results_json)

                    if "error" in search_data:
                        print(f"Error: {search_data['error']}")
                        continue
                        
                    results = search_data.get("results", [])
                    if not results:
                        print("No results found in knowledge base.")
                        continue
                    
                    # Create context from knowledge base search results
                    kb_context = "\n".join([
                        f"Document {i+1} from knowledge base:\n{result.get('content', 'No content')}\n"
                        for i, result in enumerate(results)
                    ])
                    
                    search_context += f"\n## Knowledge Base Search Results for '{query}':\n{kb_context}\n"
                
                # Process web searches
                for query in web_search_matches:
                    print(f"\n[Searching web for: '{query}']")
                    
                    try:
                        # Make sure this executes
                        print("DEBUG: Executing web search for: " + query)
                        web_result = SearchTool.search_web_free(query)
                        print(f"Web search result length: {len(web_result)}")  # Debug line
                        
                        if web_result:
                            search_context += f"\n## Web Search Results for '{query}':\n{web_result}\n"
                            
                            # Extract and add valuable information to knowledge base
                            print(f"[LEARNING] Extracting and adding information from web search to knowledge base...")
                            extracted_info = extract_and_add_search_info(client, vector_store, query, web_result)
                            learned_info.append(f"New information about '{query}' has been added to my knowledge base.")
                            
                            # Add note that the information was added to knowledge base
                            search_context += f"\n(Information from this search has been added to my knowledge base for future reference.)\n"
                        else:
                            search_context += f"\n## Web Search for '{query}' returned no results\n"
                    except Exception as e:
                        print(f"ERROR executing web search: {e}")
                        search_context += f"\n## Error executing web search for '{query}': {str(e)}\n"
                
                # Construct follow-up message with all search results
                follow_up_msg = f"""
                User question: {user_input}

                Here are the search results:
                {search_context}

                Based on these results, please provide a helpful answer to the user's question.
                Make sure to cite whether the information comes from your knowledge base or web search.
                {'' if not learned_info else 'Also, mention that: ' + ' '.join(learned_info)}
                """
                            
                # Get updated response with search results incorporated
                print("\n[Generating final response with search results...]\n")
                
                # Update messages array
                messages.pop()  # Remove previous user message
                messages.append({"role": "user", "content": follow_up_msg})
                
                # Get updated response
                updated_response = client.chat.completions.create(
                    model="gpt-4o",  # Use your actual Azure deployment name
                    messages=messages
                )
                
                final_response = updated_response.choices[0].message.content
                print(f"Assistant: {final_response}")
                
                full_response = final_response
            else:
                print(full_response)
            
            # Store assistant message
            EphemeralMemory.store_message(
                thread_id=thread_id,
                sender="assistant",
                content=full_response
            )
            
            # Update messages array for next iteration
            messages.pop()  # Remove enhanced input
            messages.append({"role": "user", "content": user_input})  # Add original user input
            messages.append({"role": "assistant", "content": full_response})  # Add assistant response
            
            # Print execution time
            elapsed_time = time.time() - start_time
            print(f"\n\n(Response generated in {elapsed_time:.2f} seconds)")
            
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Please check your Azure OpenAI configuration.")

if __name__ == "__main__":
    main()