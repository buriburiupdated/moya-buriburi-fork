"""
Agent Consultation Tool for Moya

This tool is used to provide a simple interface for agents to consult with other agents
"""

import json
from typing import Dict, List, Any, Optional, Union
from moya.registry.agent_registry import AgentRegistry
from moya.tools.base_tool import BaseTool

class AgentConsultationTool:
    """Tools for agent consultation capabilities."""
    
    @staticmethod
    def list_available_agents(agent_registry: AgentRegistry) -> str:
        """
        List all available agents that can be consulted.
        
        Args:
            agent_registry: The registry containing all available agents
            
        Returns:
            str: JSON string with list of available agents and their descriptions
        """
        agents = agent_registry.list_agents()
        agent_list = [{"name": agent_info.name, "description": getattr(agent_info, 'description', 'No description')} 
                for agent_info in agents]
        
        return json.dumps({"agents": agent_list}, indent=2)
        
    @staticmethod
    def consult_agent(agent_registry: AgentRegistry, agent_name: str, question: str, 
                      thread_id: Optional[str] = None, context: Optional[str] = None) -> str:
        """
        Consult a specific agent with a question.
        
        Args:
            agent_registry: The registry containing all available agents
            agent_name: The name of the agent to consult
            question: The question or message to send to the agent
            thread_id: Optional thread ID for conversation context
            context: Optional additional context to provide to the consulted agent
            
        Returns:
            str: The response from the consulted agent
        """
        agent = agent_registry.get_agent(agent_name)
        if agent is None:
            return json.dumps({"error": f"Agent '{agent_name}' not found."})
        
        # Format consultation message with context if provided
        consultation_message = f"[CONSULTATION] Another agent is consulting you about the following question:\n\n{question}"
        
        if context:
            consultation_message += f"\n\nAdditional context: {context}"
        
        # Create a specific thread ID for this consultation if not provided
        if not thread_id:
            consultation_thread_id = f"consultation_{agent_name}_{hash(question)}"
        else:
            consultation_thread_id = f"{thread_id}_consultation_{agent_name}"
            
        # Get response from the agent
        try:
            response = agent.handle_message(consultation_message, thread_id=consultation_thread_id)
            return json.dumps({"agent": agent_name, "response": response})
        except Exception as e:
            return json.dumps({"error": f"Error consulting {agent_name}: {str(e)}"})

    @staticmethod
    def assess_expertise(agent_registry: AgentRegistry, question: str, top_n: int = 3) -> str:
        """
        Assess which agents have expertise relevant to the given question.
        
        Args:
            agent_registry: The registry containing all available agents
            question: The question to evaluate
            top_n: Number of top agents to return
            
        Returns:
            str: JSON string with top agents and their relevance scores
        """
        agents = agent_registry.list_agents()
        scored_agents = []
        
        for agent_info in agents:
            agent = agent_registry.get_agent(agent_info.name)
            score = 0.5  # Default score
            
            # If agent has a bid_on_task method, use it to assess relevance
            if hasattr(agent, 'bid_on_task') and callable(getattr(agent, 'bid_on_task')):
                try:
                    bid_info = agent.bid_on_task(question)
                    if isinstance(bid_info, dict) and 'confidence' in bid_info:
                        score = bid_info['confidence']
                except:
                    pass
            
            scored_agents.append({
                "name": agent_info.name, 
                "description": getattr(agent_info, 'description', 'No description'),
                "relevance": score
            })
        
        # Sort by relevance score and return top N
        top_agents = sorted(scored_agents, key=lambda x: x['relevance'], reverse=True)[:top_n]
        
        return json.dumps({"query": question, "top_agents": top_agents}, indent=2)
    
    @staticmethod
    def configure_consultation_tools(tool_registry, agent_registry: AgentRegistry) -> None:
        """
        Configure consultation tools and register them with the tool registry.
        
        Args:
            tool_registry: The tool registry to register tools with
            agent_registry: The agent registry to use for consultations
        """
        # Register list available agents tool
        def list_agents_wrapper():
            return AgentConsultationTool.list_available_agents(agent_registry)
            
        tool_registry.register_tool(
            BaseTool(
                name="ListAvailableAgentsTool",
                function=list_agents_wrapper,
                description="List all available agents that can be consulted"
            )
        )
        
        # Register consult agent tool
        def consult_wrapper(agent_name, question, thread_id=None, context=None):
            return AgentConsultationTool.consult_agent(agent_registry, agent_name, question, thread_id, context)
            
        tool_registry.register_tool(
            BaseTool(
                name="ConsultAgentTool",
                function=consult_wrapper,
                description="Consult a specific agent with a question",
                parameters={
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the agent to consult",
                        "required": True
                    },
                    "question": {
                        "type": "string",
                        "description": "The question or message to send to the agent",
                        "required": True
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Optional thread ID for conversation context",
                        "required": False
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context to provide to the consulted agent",
                        "required": False
                    }
                }
            )
        )
        
        # Register assess expertise tool
        def assess_wrapper(question, top_n=3):
            return AgentConsultationTool.assess_expertise(agent_registry, question, top_n)
            
        tool_registry.register_tool(
            BaseTool(
                name="AssessAgentExpertiseTool",
                function=assess_wrapper,
                description="Assess which agents have expertise relevant to the given question",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to evaluate",
                        "required": True
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top agents to return (default: 3)",
                        "required": False
                    }
                }
            )
        )