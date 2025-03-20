"""
Interactive Dynamic RAG demo using Ollama.

This example demonstrates:
- Setting up a vector store for document retrieval
- Using Ollama for embeddings and queries
- Dynamic addition of new information to the knowledge base during chat
- Searching the internet for information not in the knowledge base
- Interactive conversation with memory persistence
- Auto-learning from web searches

Features:
- Add new information to the knowledge base with a special command
- Search for information from the knowledge base
- Fallback to web search when information isn't found locally
- Automatically add web search results to knowledge base
"""

import os
import sys
import json
import tempfile
import time
import re

from langchain_ollama import OllamaEmbeddings

from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.tool_registry import ToolRegistry
from moya.tools.rag_search_tool import VectorSearchTool
from moya.tools.search_tool import SearchTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.conversation.message import Message
from moya.conversation.thread import Thread
from moya.vectorstore.faisscpu_vectorstore import FAISSCPUVectorstoreRepository

# Import QuickTools from local example tools
from quick_tools import QuickTools


def setup_vector_store(store_path="dynamic-faiss-index"):
    """Set up or load the vector store for RAG."""
    
    # Create embeddings using Ollama
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"\nError setting up embeddings: {str(e)}")
        print("Make sure Ollama is running and the nomic-embed-text model is available.")
        print("Try: ollama pull nomic-embed-text")
        sys.exit(1)
    
    # Initialize or load the vector store
    vector_store = FAISSCPUVectorstoreRepository(store_path, embeddings)
    vector_store.create_vectorstore()
    # Create the vector store if it doesn't exist
    if not os.path.exists(store_path):
        print("Creating new vector store...")
        vector_store.create_vectorstore()
    else:
        print("Loading existing vector store...")
    
    return vector_store


def add_text_to_knowledge_base(vector_store, text, source="user_input"):
    """Add new text to the knowledge base."""
    
    # Create a temporary file to store the text
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(text)
        temp_path = temp_file.name
    
    # Add the file to the vector store
    print(f"Adding new information to knowledge base from source: {source}")
    vector_store.load_file(temp_path)
    
    # Clean up the temporary file
    os.unlink(temp_path)
    
    return True


def extract_and_add_search_info(agent, vector_store, query, search_result):
    """Extract valuable information from search results and add to knowledge base."""
    
    # Ask the agent to extract the most valuable information
    extraction_prompt = f"""
    I need you to extract the most valuable factual information from these search results related to:
    "{query}"
    
    Search results:
    {search_result[:2000]}  # Limit size to avoid context issues
    
    Please distill this into a concise, well-structured summary of the key facts and information.
    Focus only on extracting factual information, not opinions or speculative content.
    Format the information in a clear, organized way suitable for a knowledge base.
    DO NOT include any commentary, just the extracted information.
    """
    
    extracted_info = agent.handle_message(extraction_prompt)
    
    # Format the information for the knowledge base
    formatted_info = f"""
    # Information about: {query}
    
    {extracted_info}
    
    Source: Web search on {time.strftime("%Y-%m-%d")}
    Query: {query}
    """
    
    # Add to knowledge base
    add_text_to_knowledge_base(vector_store, formatted_info, source=f"web_search_{query}")
    
    return formatted_info


def perform_search(query):
    """Directly perform a web search and return the results."""
    print(f"\n[SEARCH] Searching for: {query}")
    result = SearchTool.search_web_free(query)
    print(f"[SEARCH] Search completed")
    return result


def setup_agent(vector_store):
    """Set up the Ollama agent with both RAG and search tools."""
    
    # Set up the tool registry and configure tools
    tool_registry = ToolRegistry()
    VectorSearchTool.configure_vector_search_tools(tool_registry)
    SearchTool.configure_search_tools(tool_registry)
    EphemeralMemory.configure_memory_tools(tool_registry)
    
    # Create agent configuration for Ollama with enhanced system prompt
    system_prompt = """You are a knowledgeable assistant with access to a dynamic knowledge base and web search capabilities.

    IMPORTANT INSTRUCTIONS:
    
    1. When the user asks for information, ALWAYS try the following steps in order:
    
       a. First, search your knowledge base using:
          <search query="your search query here">
       
       b. If the information is not found or incomplete in your knowledge base, search the web using:
          <free_search query="your search query here">
    
    2. When the user wants to add new information to your knowledge base, they will use the command:
       /add [information]
       
       Respond by confirming the information has been added to your knowledge base.
    
    3. Always provide helpful, accurate responses based on the available information.
    
    4. When citing information, mention the source (knowledge base or web search).
    
    5. If you're not confident in your answer even after searching, admit it rather than making up information.
    
    6. Information from web searches is automatically added to your knowledge base for future reference.
    
    Be clear, concise, and helpful in your responses.
    """

    agent_config = AgentConfig(
        agent_name="dynamic_rag_assistant",
        agent_type="ChatAgent",
        description="Ollama agent with dynamic RAG and web search capabilities",
        system_prompt=system_prompt,
        tool_registry=tool_registry,
        llm_config={
            "model_name": "llama3.1:latest",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "context_window": 4096
        }
    )
    
    # Instantiate the Ollama agent
    agent = OllamaAgent(agent_config)
    
    # Verify connection
    try:
        test_response = agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure Ollama is running and the model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1:latest")
        sys.exit(1)
    
    # Set up agent registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="dynamic_rag_assistant"
    )
    
    return orchestrator, agent


def format_conversation_context(messages):
    """Format previous messages for context."""
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    """Run the interactive dynamic RAG demo with Ollama."""
    
    print("Initializing Dynamic RAG Assistant...")
    
    # Set up vector store
    vector_store = setup_vector_store()
    
    # Test search at startup
    print("Testing search functionality...")
    test_result = SearchTool.search_web_free("test query dynamic RAG")
    print(f"Direct test search completed")
    
    # Set up agent
    orchestrator, agent = setup_agent(vector_store)
    
    # Initialize conversation thread
    thread_id = json.loads(QuickTools.get_conversation_context())["thread_id"]
    
    # Add some initial information to the knowledge base
    initial_info = """
    Dynamic RAG (Retrieval-Augmented Generation) is a system that allows adding new information 
    to a knowledge base during runtime. This enables the assistant to learn and adapt during a conversation.
    
    The key components of this system are:
    1. Vector Store - For storing and retrieving embeddings
    2. RAG Search Tool - For searching the knowledge base
    3. Web Search Tool - For finding information not in the knowledge base
    4. Dynamic Addition - For adding new information during the conversation
    5. Auto-learning - Automatically extracting and storing information from web searches
    """
    add_text_to_knowledge_base(vector_store, initial_info, source="initial_setup")
    
    print("\n" + "=" * 80)
    print("Welcome to the Dynamic RAG Assistant!")
    print("This assistant can:")
    print("- Search its knowledge base for information")
    print("- Search the web when information isn't in its knowledge base")
    print("- Learn new information that you provide")
    print("- Automatically add information from web searches to its knowledge base")
    print("\nSpecial commands:")
    print("- /add [information] - Add new information to the knowledge base")
    print("- /search [query] - Directly search the web") 
    print("- 'exit' or 'quit' - End the conversation")
    print("=" * 80)
    
    EphemeralMemory.store_message(thread_id=thread_id, sender="system", 
                                 content="Conversation initialized. Knowledge base and search capability enabled.")

    # Main conversation loop
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        # Check for add command
        if user_input.startswith("/add "):
            new_info = user_input[5:].strip()
            if new_info:
                success = add_text_to_knowledge_base(vector_store, new_info)
                
                # Store user message and system response
                EphemeralMemory.memory_repository.append_message(
                    thread_id, 
                    Message(thread_id=thread_id, sender="user", content=f"Please add this information to your knowledge base: {new_info}")
                )
                
                response = "Information successfully added to my knowledge base. I can now use this in future responses."
                EphemeralMemory.memory_repository.append_message(
                    thread_id,
                    Message(thread_id=thread_id, sender="assistant", content=response)
                )
                
                print(f"\nAssistant: {response}")
                continue
                
        # Check for direct search command
        if user_input.startswith("/search "):
            search_query = user_input[8:].strip()
            search_result = perform_search(search_query)
            
            # Auto-learn from search result
            print("\n[LEARNING] Extracting and adding information to knowledge base...")
            extracted_info = extract_and_add_search_info(agent, vector_store, search_query, search_result)
            
            print("\nDirect Search Result:")
            print(search_result[:500] + "..." if len(search_result) > 500 else search_result)
            print("\nInformation has been added to the knowledge base.")
            continue
        
        # Store user message
        EphemeralMemory.memory_repository.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
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
        
        print("\nAssistant: ", end="", flush=True)
        start_time = time.time()
        
        # Get initial response
        full_response = agent.handle_message(enhanced_input)
        
        # Check for search patterns
        vector_search_pattern = r'<search query="([^"]+)">'
        web_search_pattern = r'<free_search query="([^"]+)">'
        
        vector_search_matches = re.findall(vector_search_pattern, full_response)
        web_search_matches = re.findall(web_search_pattern, full_response)
        
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
                search_results_json = VectorSearchTool.search_vectorstore(query=query, vector_store=vector_store, k=5)
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
            
            # Process web searches - with check to avoid duplicate searches
            for query in web_search_matches:
                print(f"\n[Searching knowledge base first for: '{query}']")
                
                # First check if we already have this information in the vector store
                search_results_json = VectorSearchTool.search_vectorstore(query=query, vector_store=vector_store, k=3)
                search_data = json.loads(search_results_json)

                if "error" in search_data:
                    print(f"Error checking knowledge base: {search_data['error']}")
                    kb_has_info = False
                else:
                    results = search_data.get("results", [])
                    kb_has_info = bool(results)
                
                if kb_has_info:
                    print(f"[Found existing information in knowledge base for: '{query}']")
                    # Create context from knowledge base search results
                    kb_context = "\n".join([
                        f"Document {i+1} from knowledge base:\n{result.get('content', 'No content')}\n"
                        for i, result in enumerate(results)
                    ])
                    
                    search_context += f"\n## Knowledge Base Results for '{query}':\n{kb_context}\n"
                    search_context += f"\n(Using existing information from knowledge base; web search skipped)\n"
                else:
                    # If not in knowledge base, perform web search
                    print(f"\n[No sufficient data found in knowledge base. Searching web for: '{query}']")
                    
                    # Execute web search
                    web_result = SearchTool.search_web_free(query)
                    search_context += f"\n## Web Search Results for '{query}':\n{web_result}\n"
                    
                    # Extract and add valuable information to knowledge base
                    print(f"[LEARNING] Extracting and adding information from web search to knowledge base...")
                    extracted_info = extract_and_add_search_info(agent, vector_store, query, web_result)
                    learned_info.append(f"New information about '{query}' has been added to my knowledge base.")
                    
                    # Add note that the information was added to knowledge base
                    search_context += f"\n(Information from this search has been added to my knowledge base for future reference.)\n"
            
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
            
            # Get the response
            final_response = ""
            print("\nAssistant: ", end="", flush=True)
            
            # Try to use streaming if available, otherwise fall back to regular
            try:
                for chunk in agent.handle_message_stream(follow_up_msg):
                    print(chunk, end="", flush=True)
                    final_response += chunk
            except AttributeError:
                final_response = agent.handle_message(follow_up_msg)
                print(final_response)
                
            full_response = final_response
        else:
            print(full_response)
        
        # Store assistant message
        EphemeralMemory.memory_repository.append_message(
            thread_id,
            Message(thread_id=thread_id, sender="assistant", content=full_response)
        )
        
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\n\n(Response generated in {elapsed_time:.2f} seconds)")


if __name__ == "__main__":
    main()