"""
Quick start example for using agent consultation with Ollama models.

This example demonstrates how multiple specialized Ollama agents can consult
with each other when faced with questions outside their expertise.
"""

import os
import sys
import dotenv
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.agent_consultation_tool import AgentConsultationTool
from moya.tools.ephemeral_memory import EphemeralMemory

# Load environment variables if needed
dotenv.load_dotenv()

def setup_system():
    """Set up agents and tools for the consultation example."""
    
    # Create registries
    agent_registry = AgentRegistry()
    tool_registry = ToolRegistry()
    
    # Set up memory tools
    EphemeralMemory.configure_memory_tools(tool_registry)
    
    # Configure consultation tools - using the static method instead of instantiating
    AgentConsultationTool.configure_consultation_tools(tool_registry, agent_registry)
    
    # 1. Create math specialist agent with Ollama
    math_config = AgentConfig(
        agent_name="math_specialist",
        agent_type="ChatAgent",
        description="A specialized agent for mathematical problems and concepts",
        system_prompt="""You are a math specialist. You excel at solving mathematical problems, 
        from basic arithmetic to advanced calculus and statistics.
        
        Provide step-by-step solutions when appropriate, and explain your reasoning clearly.
        Always double-check your calculations.
        
        When consulted by other agents, focus on giving precise and accurate mathematical explanations.
        Be thorough but concise in your responses.
        """,
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )
    
    # Test Ollama connection before proceeding
    try:
        math_agent = OllamaAgent(agent_config=math_config)
        test_response = math_agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print("\nError: Make sure Ollama is running and the llama3 model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3")
        sys.exit(1)
        
    agent_registry.register_agent(math_agent)
    
    # 2. Create coding specialist agent with Ollama
    code_config = AgentConfig(
        agent_name="code_specialist",
        agent_type="ChatAgent",
        description="A specialized agent for programming and software development",
        system_prompt="""You are a coding specialist. You excel at programming, software development,
        and technical problem-solving.
        
        When providing code examples, ensure they are correct, efficient, and well-commented.
        Consider best practices and edge cases in your solutions.
        
        When consulted by other agents, focus on practical implementations and clear explanations
        of coding concepts and techniques.
        """,
        tool_registry=tool_registry,
        llm_config={
            'model_name': "codellama",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )
    
    code_agent = OllamaAgent(agent_config=code_config)
    agent_registry.register_agent(code_agent)
    
    # 3. Create main agent that can consult specialists
    main_config = AgentConfig(
        agent_name="general_assistant",
        agent_type="ChatAgent",
        description="General assistant that can consult specialists",
        system_prompt="""You are a helpful assistant who can answer general questions.
        
        For specialized questions, you can consult with expert agents using the following tools:
        - ListAvailableAgentsTool: Shows all available specialist agents
        - AssessAgentExpertiseTool: Identifies which specialists might help with a question
        - ConsultAgentTool: Consults a specific specialist with your question
        
        Available specialists:
        - math_specialist: Expert in mathematics from basic to advanced
        - code_specialist: Expert in programming and software development
        
        When faced with a technical question in math or programming, consider consulting the
        appropriate specialist using these tools.
        
        To use these tools:
        1. First call ListAvailableAgentsTool to confirm available specialists
        2. Then use AssessAgentExpertiseTool with the question to identify which specialists might help
        3. Finally call ConsultAgentTool with the agent_name and question
        
        After receiving their response, synthesize it into your answer to provide the best help to the user.
        
        Always cite the specialist when you use their input in your response.
        """,
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )
    
    main_agent = OllamaAgent(agent_config=main_config)
    agent_registry.register_agent(main_agent)
    
    # Create orchestrator with main agent as default
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="general_assistant"
    )
    
    return orchestrator

def main():
    """Run the interactive agent consultation example."""
    print("Starting Ollama Agent Consultation example...")
    print("This example requires Ollama to be running with appropriate models.")
    print("Make sure you have started Ollama with the required models installed.")
    print("\nInitializing system...")
    
    try:
        orchestrator = setup_system()
        print("System initialized successfully!")
    except Exception as e:
        print(f"Error setting up system: {str(e)}")
        print("Make sure Ollama is running with the required models.")
        return
    
    # For conversation history
    thread_id = "ollama_consultation_example"
    print("\nWelcome to the Ollama Agent Consultation Example")
    print("Ask any question. The main agent will consult specialists for math or coding questions.")
    print("Type 'exit' to quit.\n")
    
    def stream_callback(chunk):
        """Print response chunks as they arrive."""
        print(chunk, end="", flush=True)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        
        # Store user message using EphemeralMemory static method directly
        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="user",
            content=user_input
        )
        
        # Process with orchestrator
        print("\nAssistant: ", end="", flush=True)
        response = orchestrator.orchestrate(
            thread_id=thread_id,
            user_message=user_input,
            stream_callback=stream_callback
        )
        
        # Store assistant response using EphemeralMemory static method directly
        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="assistant",
            content=response
        )

if __name__ == "__main__":
    main()