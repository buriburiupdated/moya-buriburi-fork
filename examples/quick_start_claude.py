"""
Interactive chat example using Claude agent with conversation memory.
"""

import os
import dotenv
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.claude_agent import ClaudeAgent, ClaudeAgentConfig
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.memory.file_system_repo import FileSystemRepository
import json
from quick_tools import QuickTools
from moya.tools.base_tool import BaseTool
dotenv.load_dotenv()


def setup_agent():
    # Set up memory components
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    tool_registry.register_tool(BaseTool(name="ConversationContext", function=QuickTools.get_conversation_context))

    config = ClaudeAgentConfig(
        agent_name="claude_agent",
        description="An interactive chat agent using Claude",
        api_key=os.getenv("CLAUDE_KEY"),
        model_name="claude-3-opus-20240229",
        agent_type="ChatAgent",
        tool_registry=tool_registry,
        is_streaming=True,
        system_prompt="You are an interactive chat agent that can remember previous conversations. "
                      "You have access to tools that help you to store and retrieve conversation history. "
                      "Use the conversation history for your reference in answering any user query. "
                      "Be helpful and polite in your responses, and be concise and clear. "
                      "Be useful but do not provide any information unless asked.",
    )

    # Create Claude agent with memory capabilities
    agent = ClaudeAgent(config)

    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="claude_agent"
    )

    return orchestrator, agent


def main():
    orchestrator, agent = setup_agent()
    thread_id = json.loads(QuickTools.get_conversation_context())["thread_id"]

    print("Welcome to Interactive Chat! (Type 'quit' or 'exit' to end)")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store user message
        EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=user_input)
    
        session_summary = EphemeralMemory.get_thread_summary(thread_id)
        enriched_input = f"{session_summary}\nCurrent user message: {user_input}"

        # Print Assistant prompt
        print("\nAssistant: ", end="", flush=True)

        # Define callback for streaming
        def stream_callback(chunk):
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                print(chunk.delta.text, end="", flush=True)

        # Get response using stream_callback and thread_id
        try:
            response = agent.handle_message_stream(
                message=enriched_input,
                stream_callback=stream_callback,
                thread_id=thread_id
            )
            if not response:
                print("I apologize, but I'm having trouble generating a response.")
                response = "Error generating response"
        except Exception as e:
            print(f"\nError: {e}")
            response = "Error occurred during conversation"

        EphemeralMemory.store_message(thread_id=thread_id, sender="assistant", content=response)
        print()


if __name__ == "__main__":
    main()