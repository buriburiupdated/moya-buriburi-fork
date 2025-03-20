"""
Interactive chat example using Ollama agent with custom user-defined tools.
"""

import sys
from moya.tools.tool_registry import ToolRegistry
from moya.tools.dynamic_tool_registrar import DynamicToolRegistrar
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.ollama_agent import OllamaAgent, AgentConfig


# User-defined functions to be registered as tools
def temperature_converter(celsius: float = None, fahrenheit: float = None) -> dict:
    """
    Convert temperatures between Celsius and Fahrenheit.
    
    Parameters:
    - celsius: Temperature in Celsius to convert to Fahrenheit
    - fahrenheit: Temperature in Fahrenheit to convert to Celsius
    
    Returns:
    - Dictionary with converted temperatures
    """
    result = {}
    
    if celsius is not None:
        result["fahrenheit"] = (celsius * 9/5) + 32
    
    if fahrenheit is not None:
        result["celsius"] = (fahrenheit - 32) * 5/9
    
    return result


def generate_password(length: int = 12, include_special: bool = True) -> str:
    """
    Generate a random password.
    
    Parameters:
    - length: Length of the password (default: 12)
    - include_special: Whether to include special characters (default: True)
    
    Returns:
    - Generated password
    """
    import random
    import string
    
    chars = string.ascii_letters + string.digits
    if include_special:
        chars += string.punctuation
    
    return ''.join(random.choice(chars) for _ in range(length))


def setup_agent():
    """Set up Ollama agent with custom user-defined tools."""
    # Create a tool registry
    tool_registry = ToolRegistry()
    
    # Register user functions as tools
    DynamicToolRegistrar.register_function_as_tool(
        tool_registry=tool_registry,
        function=temperature_converter
    )
    
    DynamicToolRegistrar.register_function_as_tool(
        tool_registry=tool_registry,
        function=generate_password,
        name="PasswordGenerator",  # Custom name (optional)
        description="Generates secure random passwords"  # Custom description (optional)
    )
    
    # Setup an agent with these tools
    config = AgentConfig(
        agent_name="custom_tools_agent",
        agent_type="OllamaAgent",
        description="An agent with user-defined tools",
        system_prompt="You are a helpful assistant with access to custom tools for temperature conversion and password generation. Be concise and clear in your responses.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )
    
    # Create the agent
    try:
        agent = OllamaAgent(agent_config=config)
        
        # Verify Ollama connection with simple test request
        test_response = agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print("\nError: Make sure Ollama is running and the llama3.1 model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1")
        sys.exit(1)
    
    # Register agent and create orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="custom_tools_agent"
    )
    
    return orchestrator, agent


def main():
    """Run interactive chat with the Ollama agent using custom tools."""
    print("\n🛠️  Custom Tools with Ollama Chat 🛠️")
    print("--------------------------------------")
    print("This example demonstrates creating custom tools from user-defined functions.")
    print("Try these examples:")
    print("- Convert 32 Celsius to Fahrenheit")
    print("- What is 68 Fahrenheit in Celsius?")
    print("- Generate a secure password with 20 characters")
    print("- Make me a password without special characters")
    print("\nType 'exit' to quit.\n")
    
    # Set up the agent and orchestrator
    orchestrator, _ = setup_agent()
    
    # Create a thread ID for the conversation
    thread_id = "custom_tools_demo"
    
    # Interactive chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! 👋")
            break
        
        # Process user input through the orchestrator
        print("\nAgent is thinking...")
        response = orchestrator.orchestrate(thread_id=thread_id, user_message=user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()