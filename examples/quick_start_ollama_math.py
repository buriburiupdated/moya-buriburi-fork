"""
Math Solver demo with Ollama and enhanced MathTool.

This example demonstrates:
- Setting up an Ollama agent with enhanced math capabilities
- Solving various types of math problems
- Interactive mathematics problem solving
"""

import sys
from moya.tools.tool_registry import ToolRegistry
from moya.tools.math_tool import MathTool
from moya.agents.base_agent import AgentConfig
from moya.agents.ollama_agent import OllamaAgent
from moya.conversation.message import Message
from moya.conversation.thread import Thread
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.ephemeral_memory import EphemeralMemory


def setup_math_agent():
    """Set up an Ollama agent with enhanced math capabilities."""
    # Set up the tool registry and configure math tools
    tool_registry = ToolRegistry()
    MathTool.configure_math_tools(tool_registry)
    EphemeralMemory.configure_memory_tools(tool_registry)
    
    # Create a math-focused agent with Ollama
    math_agent_config = AgentConfig(
        agent_name="math_wizard",
        agent_type="ChatAgent",
        description="A specialized mathematical assistant that can solve equations, compute derivatives and integrals, evaluate expressions, solve systems of equations, compute limits, and expand series.",
        system_prompt=(
            "You are a mathematics expert assistant with advanced symbolic math capabilities. "
            "Use the provided math tools to solve problems precisely and show your work step-by-step. "
            "For complex problems, break them down into smaller parts that can be solved with the available tools. "
            "Always verify your solutions when possible."
        ),
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'temperature': 0.2,  # Lower temperature for precise math
            'base_url': "http://localhost:11434",
            'context_window': 4096
        }
    )

    # Create the agent
    math_agent = OllamaAgent(math_agent_config)

    # Verify Ollama connection
    try:
        test_response = math_agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print("\nError: Make sure Ollama is running and the model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1:latest")
        sys.exit(1)

    return math_agent


def format_conversation_context(messages):
    """Format previous messages for context."""
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    """Run the interactive math solver demo with Ollama."""
    math_agent = setup_math_agent()
    thread_id = "math_solver_demo_001"

    print("\n" + "=" * 65)
    print("Welcome to the Advanced Math Solver Demo with Ollama!")
    print("This assistant can help with calculus, algebra, and other math problems.")
    print("Examples of what you can ask:")
    print("  - Solve the equation x^2 + 3x + 2 = 0")
    print("  - Find the derivative of sin(x^2)")
    print("  - Integrate x^3 with respect to x")
    print("  - Solve the system of equations: x + y = 10, 2x - y = 5")
    print("  - Find the limit of sin(x)/x as x approaches 0")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 65)

    # Setup memory for conversation history
    memory_repo = InMemoryRepository()
    memory_repo.create_thread(Thread(thread_id=thread_id))

    def print_streaming_response(chunk):
        """Callback for streaming responses."""
        print(chunk, end="", flush=True)

    while True:
        user_input = input("\n\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye! Hope you enjoyed solving math problems!")
            break

        # Store user message
        memory_repo.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
        )

        # Get conversation context
        thread = memory_repo.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)

        if previous_messages and len(previous_messages) > 1:
            context = format_conversation_context(previous_messages)
            enhanced_input = f"{context}\nCurrent query: {user_input}"
        else:
            enhanced_input = user_input

        try:
            print("\nMath Wizard: ", end="", flush=True)
            
            # Get response from the math agent with streaming
            response = ""
            message_stream = math_agent.handle_message_stream(enhanced_input, thread_id=thread_id)
            
            if message_stream:
                for chunk in message_stream:
                    print_streaming_response(chunk)
                    response += chunk
            else:
                response = math_agent.handle_message(enhanced_input, thread_id=thread_id)
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