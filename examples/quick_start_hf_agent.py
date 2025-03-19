"""
Interactive chat example using a HuggingFace agent.

This example demonstrates:
- Setting up a single HuggingFace agent with TinyLlama
- Integrating tools with the HuggingFace model
- Streaming responses
"""

import sys
import os
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.tool_registry import ToolRegistry
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.tools.math_tool import MathTool
from moya.agents.huggingface_agent import HuggingFaceAgent, HuggingFaceAgentConfig
from moya.conversation.message import Message
from moya.conversation.thread import Thread


def setup_agent():
    """Set up a single HuggingFace agent."""
    # Set up tools
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    MathTool.configure_math_tools(tool_registry)
    
    # Add a local small model - good for testing
    try:
        tiny_llama_config = HuggingFaceAgentConfig(
            agent_name="tiny_llama",
            agent_type="HuggingFaceAgent",
            description="A small but capable assistant powered by TinyLlama",
            system_prompt="You are a helpful AI assistant that provides concise and accurate information. You can solve math problems using the provided tools.",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            task="text-generation",
            tool_registry=tool_registry,
            use_api=False,
            device="cpu",  # Use "cuda:0" for GPU
            generation_config={
                'max_new_tokens': 512,
                'temperature': 0.7,
                'do_sample': True
            }
        )
        agent = HuggingFaceAgent(tiny_llama_config)
        print("TinyLlama model loaded successfully")
        return agent
    except Exception as e:
        print(f"Error loading TinyLlama: {str(e)}")
        sys.exit(1)


def format_conversation_context(messages):
    """Format previous messages for context."""
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    """Run the interactive chat demo with a HuggingFace model."""
    print("\nSetting up HuggingFace model...")
    agent = setup_agent()
    thread_id = "huggingface_chat_001"

    print("\n" + "=" * 70)
    print("Welcome to the HuggingFace TinyLlama Demo!")
    print("This example shows how to use TinyLlama with the Moya framework.")
    print("Type 'quit' or 'exit' to end the chat.")
    print("=" * 70)

    # Set up memory
    memory_repo = InMemoryRepository()
    memory_repo.create_thread(Thread(thread_id=thread_id))

    def print_streaming_response(chunk):
        """Callback for streaming responses."""
        print(chunk, end="", flush=True)

    while True:
        user_input = input("\n\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store user message
        memory_repo.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
        )

        # Get conversation context
        thread = memory_repo.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)

        if previous_messages:
            context = format_conversation_context(previous_messages)
            enhanced_input = f"{context}\nCurrent user message: {user_input}"
        else:
            enhanced_input = user_input

        try:
            print("\nAssistant: ", end="", flush=True)
            
            # Use streaming response
            response = ""
            message_stream = agent.handle_message_stream(enhanced_input, thread_id=thread_id)
            
            if message_stream:
                for chunk in message_stream:
                    print_streaming_response(chunk)
                    response += chunk
            else:
                # Fallback to non-streaming if stream is not available
                response = agent.handle_message(enhanced_input, thread_id=thread_id)
                print(response)

            # Store assistant's response
            memory_repo.append_message(
                thread_id,
                Message(thread_id=thread_id, sender="assistant", content=response)
            )
            
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()