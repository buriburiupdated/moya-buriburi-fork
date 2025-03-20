"""
Interactive chat example using DeepSeek agent with conversation memory.
"""

import os
import sys
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.tool_registry import ToolRegistry
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.base_agent import AgentConfig
from moya.conversation.message import Message
from moya.conversation.thread import Thread
from moya.agents.deepseek_agent import DeepSeekAgent


def setup_agent():
    # Set up memory components
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)

    # Get API key from environment variable
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    if not api_key:
        print("\nError: DeepSeek API key not found in environment variables.")
        print("Set your API key: export DEEPSEEK_API_KEY=your_api_key_here")
        sys.exit(1)

    # Create DeepSeek agent with memory capabilities
    agent_config = AgentConfig(
        agent_name="deepseek-coder",
        agent_type="ChatAgent",
        description="An AI assistant powered by DeepSeek with memory",
        system_prompt="You are a helpful AI assistant. Be concise and clear.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "deepseek-chat",  # Using available model from test results
            'temperature': 0.7,
            'api_key': "",  # API key is pulled from environment variable
            'context_window': 8192
        }
    )

    agent = DeepSeekAgent(agent_config)

    # Verify DeepSeek connection
    try:
        test_response = agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from DeepSeek test query")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure DeepSeek API key is set correctly and you have sufficient credits.")
        sys.exit(1)

    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="deepseek_assistant"
    )

    return orchestrator, agent


def format_conversation_context(messages):
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    orchestrator, agent = setup_agent()
    thread_id = "interactive_chat_001"

    print("Welcome to Interactive Chat! (Type 'quit' or 'exit' to end)")
    print("-" * 50)

    session_memory = EphemeralMemory.memory_repository
    session_memory.create_thread(Thread(thread_id=thread_id))

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store user message
        session_memory.append_message(thread_id, Message(thread_id=thread_id, sender="user", content=user_input))

        # Get conversation context
        thread = session_memory.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)

        if previous_messages:
            context = format_conversation_context(previous_messages)
            enhanced_input = f"{context}\nCurrent user message: {user_input}"
        else:
            enhanced_input = user_input

        try:
            print("\nAssistant: ", end="", flush=True)

            response = ""
            try:
                # Use enhanced_input instead of user_input for context
                for chunk in agent.handle_message_stream(enhanced_input):
                    if chunk:
                        print(chunk, end="", flush=True)
                        response += chunk
            except Exception as e:
                # Fallback to non-streaming with enhanced input
                response = agent.handle_message(enhanced_input)
                if response:
                    print(response)

            print()

            if not response or response.startswith("[DeepSeekAgent error"):
                print("\nError: No response received. Please try again.")
                continue

            # Store the assistant's response
            session_memory.append_message(thread_id, Message(thread_id=thread_id, sender="assistant", content=response))

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}. Please try again.")
            continue


if __name__ == "__main__":
    main()