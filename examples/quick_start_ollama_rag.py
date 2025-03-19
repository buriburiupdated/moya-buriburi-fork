"""
Interactive RAG demo that vectorizes docs folder HTML files and uses Ollama to answer questions
about the Moya framework with proper document citations.
"""

import os
import sys
import glob
import json
import re

from langchain_ollama import OllamaEmbeddings

from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.tool_registry import ToolRegistry
from moya.tools.rag_search_tool import VectorSearchTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.conversation.message import Message
from moya.conversation.thread import Thread
from moya.vectorstore.faisscpu_vectorstore import FAISSCPUVectorstoreRepository

from quick_tools import QuickTools


def setup_agent(vector_store):
    """Set up the Ollama agent with RAG search tool."""
    # Set up the tool registry and configure tools
    tool_registry = ToolRegistry()
    VectorSearchTool.configure_vector_search_tools(tool_registry)
    # VectorSearchTool.configure_vector_search_tools(tool_registry, vector_store=vector_store)
    EphemeralMemory.configure_memory_tools(tool_registry)
    
    # Create agent configuration for Ollama
    system_prompt = """You are a knowledgeable assistant for the Moya multi-agent framework.

    IMPORTANT: For ANY question about Moya, you MUST search the documentation first.
    DO NOT rely on your general knowledge about Moya. Instead, follow these steps for every response:

    1. Search the documentation using this exact format:
       <search query="your search query here">
       
       For example: <search query="What is Moya?">

    2. After receiving search results, analyze them carefully.

    3. Provide your answer based on the search results.

    4. Always cite sources by mentioning which file the information comes from.

    Be clear, concise, and helpful in your responses.
    """

    agent_config = AgentConfig(
        agent_name="moya_docs_assistant",
        agent_type="ChatAgent",
        description="Ollama agent integrated with RAG search of Moya documentation",
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
        print("\nError: Make sure Ollama is running and the model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1:latest")
        sys.exit(1)
    
    # Set up agent registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="moya_docs_assistant"
    )
    
    return orchestrator, agent

def format_conversation_context(messages) -> str:
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    docs_dir = "docs"
    path = "faiss-index"
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISSCPUVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()

    print("Hello! Initializing the Moya Documentation Assistant...")
    # Add documents to vector store
    print(os.path.abspath(docs_dir))
    for file in glob.glob(os.path.join(docs_dir, "*")):
        print(f"Adding document: {file}")
        vector_store.load_file(file)
    
    print("All documents added to vector store.")
    # Set up agent with RAG search tool
    orchestrator, agent = setup_agent(vector_store)
    thread_id = json.loads(QuickTools.get_conversation_context())["thread_id"]
    
    print("\n" + "=" * 80)
    print("Welcome to the Moya Documentation Assistant!")
    print("Ask questions about the Moya framework and get answers from the documentation.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 80)
    
    # Example questions to help users get started
    print("\nExample questions you can ask:")
    print("- What is Moya?")
    print("- How do I create an agent?")
    print("- What types of agents are supported?")
    print("- How does memory management work?")
    print("- What is tool registry?")

    EphemeralMemory.store_message(thread_id=thread_id, sender="system", 
                                  content="Conversation initialized. Search capability is enabled.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        # Store user message
        EphemeralMemory.memory_repository.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
        )
        
        # Retrieve conversation context
        thread = EphemeralMemory.memory_repository.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)
        context = format_conversation_context(previous_messages) if previous_messages else ""
        
        # Enhanced input with instruction to use RAG search
        enhanced_input = (
            f"{context}"
            f"User: {user_input}\n\n"
            f"Remember to use the VectorSearchTool to search for information in the documentation."
        )
        
        print("\nAssistant: ", end="", flush=True)
        
        # Get initial response
        full_response = agent.handle_message(enhanced_input)
        
        # Check for search tag pattern
        search_pattern = r'<search query="([^"]+)">'
        search_matches = re.findall(search_pattern, full_response)
        
        # If tool calls were made, process them and regenerate response
        if search_matches:
            # Clear the previous output
            if os.name == 'posix':  # For Linux/Mac
                    print("\033[F\033[K" * (full_response.count('\n') + 1), end="")
            else:  # For Windows
                os.system('cls' if os.name == 'nt' else 'clear')

            print(f"\nYou: {user_input}")
            
            # Process each search query
            for query in search_matches:
                print(f"\n[Searching documentation for: '{query}']")
                
                # Execute vector search
                search_results_json = VectorSearchTool.search_vectorstore(query=query, vector_store=vector_store, k=5)
                search_data = json.loads(search_results_json)

                # Print search results in readable format
                if "error" in search_data:
                    print(f"Error: {search_data['error']}")
                    continue
                    
                results = search_data.get("results", [])
                if not results:
                    print("No results found.")
                    continue
                    
                for i, result in enumerate(results):
                    content = result.get("content", "No content")
                    content_preview = content[:150] + "..." if len(content) > 150 else content
                    metadata = result.get("metadata", {})
                    source = metadata.get("source", "Unknown")
                
                # Create context from search results
                search_context = "\n".join([
                    f"Document {i+1} (Source: {result.get('metadata', {}).get('source', 'Unknown')}):\n{result.get('content', 'No content')}\n"
                    for i, result in enumerate(results)
                ])
                
                # Construct follow-up message with search results
                follow_up_msg = f"""
                User question: {user_input}

                Here are relevant documents from the Moya documentation:

                {search_context}

                Based on these documents, please provide a helpful answer to the user's question.
                Make sure to cite the sources of information.
                """
                                
                # Get updated response with search results incorporated
                print("\n[Generating final response with search results...]\n")
                updated_response = agent.handle_message_stream(follow_up_msg)
                
                full_response = ""
                print("\nAssistant: ", end="", flush=True)
                for chunk in updated_response:
                    print(chunk, end="", flush=True)
                    full_response += chunk
        else:
            print(full_response)
        
        # Store response in memory
        EphemeralMemory.memory_repository.append_message(
            thread_id,
            Message(thread_id=thread_id, sender="assistant", content=full_response)
        )
        
        print()  # Add a newline after the response


if __name__ == "__main__":
    main()