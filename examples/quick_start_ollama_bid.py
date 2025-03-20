"""
Interactive chat example using BidOrchestrator with multiple Ollama agents.

This example demonstrates:
- Setting up multiple specialized agents with Ollama
- Using the BidOrchestrator to dynamically select the best agent for each request
- Team formation for complex queries
- Interactive chat with conversation memory
"""

import sys
import time
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.tool_registry import ToolRegistry
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.bid_orchestrator import BidOrchestrator
from moya.agents.base_agent import AgentConfig
from moya.agents.ollama_agent import OllamaAgent
from moya.conversation.message import Message
from moya.conversation.thread import Thread


def setup_agents():
    """Set up multiple specialized agents and configure the BidOrchestrator."""
    # Set up memory components
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)

    # Create three specialized agents
    code_agent_config = AgentConfig(
        agent_name="code_assistant",
        agent_type="ChatAgent",
        description="An expert in programming and software development. Specializes in writing code, debugging, and explaining technical concepts.",
        system_prompt="You are an AI coding assistant. Provide accurate, efficient code solutions with explanations. Be concise and focus on best practices.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'temperature': 0.3,  # Lower temperature for code
            'base_url': "http://localhost:11434",
            'context_window': 4096
        }
    )

    general_agent_config = AgentConfig(
        agent_name="general_assistant",
        agent_type="ChatAgent",
        description="A helpful general knowledge assistant that can answer questions on a wide range of topics.",
        system_prompt="You are a helpful AI assistant with broad knowledge. Answer questions accurately and be conversational but concise.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'temperature': 0.7,
            'base_url': "http://localhost:11434",
            'context_window': 4096
        }
    )

    creative_agent_config = AgentConfig(
        agent_name="creative_assistant",
        agent_type="ChatAgent",
        description="A creative assistant specializing in writing, storytelling, and creative content generation.",
        system_prompt="You are a creative AI assistant. Generate imaginative content, stories, and ideas. Be engaging and original in your responses.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'temperature': 0.9,  # Higher temperature for creativity
            'base_url': "http://localhost:11434",
            'context_window': 4096
        }
    )

    # Create the agents
    code_agent = OllamaAgent(code_agent_config)
    general_agent = OllamaAgent(general_agent_config)
    creative_agent = OllamaAgent(creative_agent_config)

    # Verify Ollama connection
    try:
        test_response = general_agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print("\nError: Make sure Ollama is running and the model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1:latest")
        sys.exit(1)

    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(code_agent)
    agent_registry.register_agent(general_agent)
    agent_registry.register_agent(creative_agent)
    
    # Configure the BidOrchestrator
    orchestrator = BidOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="general_assistant",
        config={
            "parallel_bidding": True,
            "min_confidence": 0.6,
            "team_threshold": 0.6,
            "max_team_size": 2,
            "enable_learning": True,
            "verbose": True  # Set to True to see orchestration details
        }
    )

    return orchestrator


def format_conversation_context(messages):
    """Format previous messages for context."""
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    """Run the interactive chat demo with the BidOrchestrator."""
    orchestrator = setup_agents()
    thread_id = "bid_orchestrator_chat_001"

    print("\n" + "=" * 50)
    print("Welcome to BidOrchestrator Demo with Ollama!")
    print("This example demonstrates how agents bid on and handle your queries.")
    print("Type 'quit' or 'exit' to end the chat.")
    print("=" * 50)

    session_memory = EphemeralMemory.memory_repository
    session_memory.create_thread(Thread(thread_id=thread_id))

    def print_streaming_response(chunk):
        """Callback for streaming responses."""
        print(chunk, end="", flush=True)

    while True:
        user_input = input("\n\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store user message
        session_memory.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
        )

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
            
            # Use the orchestrator to handle the message with streaming
            response = orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=enhanced_input,
                stream_callback=print_streaming_response
            )

            # Store assistant's response
            session_memory.append_message(
                thread_id,
                Message(thread_id=thread_id, sender="assistant", content=response)
            )
            
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()