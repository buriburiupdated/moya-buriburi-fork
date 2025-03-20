"""
Interactive chat example using Claude agent with web search capability.
"""

import os
import dotenv
import json
import re  # Add this import
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.claude_agent import ClaudeAgent, ClaudeAgentConfig
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.tools.search_tool import SearchTool
from quick_tools import QuickTools
from moya.tools.base_tool import BaseTool

dotenv.load_dotenv()

def setup_agent():
    # Set up memory components and search tools
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    SearchTool.configure_search_tools(tool_registry)
    tool_registry.register_tool(BaseTool(name="ConversationContext", function=QuickTools.get_conversation_context))

    config = ClaudeAgentConfig(
        agent_name="claude_search_agent",
        description="An interactive chat agent using Claude with web search capability",
        api_key=os.getenv("CLAUDE_KEY"),
        model_name="claude-3-opus-20240229",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        is_streaming=True,
        system_prompt="""You are an interactive chat agent that can remember previous conversations and search the web.
        
You have access to the following important tools:
1. Memory tools - Use these to store and retrieve conversation history
2. WebSearchTool - Use this to search the web for current information using SerpAPI (requires API key)
3. FreeWebSearchTool - Use this to search the web using DuckDuckGo (no API key required)

IMPORTANT: To use the search tools, you MUST use the following format exactly:
<search query="your search query here">

CRITICAL INSTRUCTION: Your training data is outdated. It's currently 2025. ALWAYS TRUST the search results 
over your training data when answering questions about current events, people in office, or other factual information.
The search results represent the current reality, not fictional or speculative information.

Instructions for using web search:
- When asked about recent events, facts you're uncertain about, or current information, use a web search tool
- Begin by using FreeWebSearchTool first as it doesn't require an API key
- Fall back to WebSearchTool only if the free tool doesn't provide adequate results
- Always cite your sources when providing information from search results
- Format citations as: [Source: Title of page]

Use the conversation history for your reference in answering user queries.
Be helpful, polite, concise and clear in your responses.
""",
    )

    # Create Claude agent with memory and search capabilities
    agent = ClaudeAgent(config)

    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="claude_search_agent"
    )

    return orchestrator, agent

# Add a direct search function
def perform_search(query):
    """Directly perform a web search and return the results."""
    print(f"\n[DIRECT SEARCH] Searching for: {query}")
    result = SearchTool.search_web_free(query)
    print(f"[DIRECT SEARCH] Search completed")
    return result


def main():
    # Test search at startup
    print("Testing search functionality...")
    test_result = SearchTool.search_web_free("test query moya framework")
    print(f"Direct test search result completed")
    
    orchestrator, agent = setup_agent()
    thread_id = json.loads(QuickTools.get_conversation_context())["thread_id"]

    print("Welcome to Claude Search Agent! (Type 'quit' or 'exit' to end)")
    print("Try asking questions that might benefit from web search!")
    print("-" * 50)

    # Store a system message to initialize the thread
    EphemeralMemory.store_message(thread_id=thread_id, sender="system", 
                                  content="Conversation initialized. Search capability is enabled.")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
            
        # Add direct search command
        if user_input.startswith("!search "):
            search_query = user_input[8:].trip()
            search_result = perform_search(search_query)
            print("\nDirect Search Result:")
            print(search_result[:500] + "..." if len(search_result) > 500 else search_result)
            continue

        # Store user message
        EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=user_input)
    
        session_summary = EphemeralMemory.get_thread_summary(thread_id)
        enriched_input = f"{session_summary}\nCurrent user message: {user_input}"

        # Print Assistant prompt
        print("\nAssistant: ", end="", flush=True)

        # Accumulate full response to check for search patterns
        full_response = ""
        
        # Define callback for streaming
        def stream_callback(chunk):
            nonlocal full_response
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                text = chunk.delta.text
                full_response += text
                print(text, end="", flush=True)

        # Get response using stream_callback and thread_id
        try:
            response = agent.handle_message_stream(
                message=enriched_input,
                stream_callback=stream_callback,
                thread_id=thread_id
            )
            
            # Process search patterns
            search_pattern = r'<search query="([^"]+)">'
            search_matches = re.findall(search_pattern, full_response)

            # If searches were performed, we'll replace the original response
            if search_matches:
                # Clear the console output from the original response
                if os.name == 'posix':  # For Linux/Mac
                    print("\033[F\033[K" * (full_response.count('\n') + 1), end="")
                else:  # For Windows
                    os.system('cls' if os.name == 'nt' else 'clear')
                
                # Reprint the user input
                print(f"\nYou: {user_input}")
                
                updated_responses = []
                for query in search_matches:
                    # Only print the search section starting with [DIRECT SEARCH]
                    search_result = perform_search(query)
                    
                    # Get updated response silently
                    follow_up_msg = f"""Here are the accurate, up-to-date search results for '{query}':
{search_result}

IMPORTANT: These search results represent the current reality as of 2025.
Please provide an accurate response based ENTIRELY on these search results.
Give only the concise factual response without any prefixes or qualifications.
"""
                    updated_response = agent.handle_message(follow_up_msg, thread_id=thread_id)
                    updated_responses.append(updated_response)
                
                # Print only the final response
                final_response = updated_responses[-1]
                print("\nAssistant: " + final_response)

            # If no searches were needed, just use the original response
            else:
                response = full_response

            if not response:
                print("I apologize, but I'm having trouble generating a response.")
                response = "Error generating response"
        except Exception as e:
            print(f"\nError: {e}")
            response = f"Error occurred during conversation: {str(e)}"

        # Store only the final response
        EphemeralMemory.store_message(thread_id=thread_id, sender="assistant", content=response)
        print()


if __name__ == "__main__":
    main()