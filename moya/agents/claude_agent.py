"""
ClaudeAgent for Moya.

An Agent that uses Claude's ChatCompletion or Completion API
to generate responses, pulling API key from the environment.
"""

import os
from anthropic import Anthropic
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from moya.agents.base_agent import Agent
from moya.agents.base_agent import AgentConfig
from moya.tools.base_tool import BaseTool
from moya.tools.tool_registry import ToolRegistry
from moya.memory.base_repository import BaseMemoryRepository

@dataclass
class ClaudeAgentConfig(AgentConfig):
    """
    Configuration data for a ClaudeAgent.
    """
    model_name: str = "claude-3-opus-20240229"
    api_key: str = None
    tool_choice: Optional[str] = None

class ClaudeAgent(Agent):
    """
    A Claude-based agent that uses the Anthropic API.
    """

    def __init__(
        self,
        config: ClaudeAgentConfig   
    ):
        """
        Initialize the ClaudeAgent.

        :param config: Configuration for the agent.
        """
        super().__init__(config=config)
        self.model_name = config.model_name
        if not config.api_key:
            raise ValueError("Anthropic API key is required for ClaudeAgent.")
        self.client = Anthropic(api_key=config.api_key)
        self.system_prompt = config.system_prompt
        self.tool_choice = config.tool_choice if config.tool_choice else None
        self.max_iterations = 5

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Discover tools available for this agent.
        """
        if not self.tool_registry:
            return None
        
        # Generate tool definitions for Claude API format
        tools = []
        for tool in self.tool_registry.get_tools():
            properties = {}
            required = []
            
            for name, info in tool.parameters.items():
                properties[name] = {"type": "string", "description": info["description"]}
                if info.get("required", False):
                    required.append(name)
            
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
        return tools

    def handle_message(self, message: str, **kwargs) -> str:
        """
        Calls Claude API to handle the user's message.
        """
        return self.handle(message)

    def handle_message_stream(self, message: str, stream_callback=None, **kwargs):
        """
        Calls Claude API to handle the user's message with streaming support.
        """
        thread_id = kwargs.get('thread_id')
        return self.handle_stream(message, stream_callback, thread_id=thread_id)

    def handle_stream(self, user_message, stream_callback=None, thread_id=None):
        """
        Handle a chat session with streaming support.
        """
        conversation = [{"role": "user", "content": user_message}]

        # Get message and handle streaming
        message = self.get_response(conversation, stream_callback)
        return message.get("content", "")

    def get_response(self, conversation, stream_callback=None):
        """Generate a response via the Claude API with streaming support."""
        messages = []
        for msg in conversation:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})

        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4096,
        }

        if self.system_prompt:
            params["system"] = self.system_prompt

        try:
            if stream_callback and self.is_streaming:
                response = self.client.messages.create(**params, stream=True)
                response_text = ""
                for chunk in response:
                    # Handle message content events
                    if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            text = chunk.delta.text
                            response_text += text
                            if stream_callback:
                                stream_callback(chunk)
                return {"content": response_text, "tool_calls": []}
            else:
                response = self.client.messages.create(**params)
                return {"content": response.content[0].text, "tool_calls": []}
                
        except Exception as e:
            print(f"Error in API call: {e}")
            return {"content": "I encountered an error processing your request.", "tool_calls": []}

    def handle(self, user_message):
        """
        Handle a chat session with the user and resolve tool calls iteratively.
        
        Args:
            user_message (str): The initial message from the user.
        
        Returns:
            str: Final response after tool call processing.
        """
        conversation = [{"role": "user", "content": user_message}]
        iteration = 0

        while iteration < self.max_iterations:
            message = self.get_response(conversation)
            # Extract message content
            if isinstance(message, dict):
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
            else:
                content = message.content if message.content is not None else ""
                tool_calls = message.tool_calls if hasattr(message, "tool_calls") and message.tool_calls else []
                # Convert to list of dicts if it's not already
                if tool_calls and not isinstance(tool_calls[0], dict):
                    tool_calls = [tc.dict() for tc in tool_calls]
                    
            # Create assistant message entry
            entry = {"role": "assistant", "content": content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            conversation.append(entry)

            # Process tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    tool_response = self.handle_tool_call(tool_call)
                    
                    conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": tool_response
                    })
                iteration += 1
            else:
                break

        final_message = conversation[-1].get("content", "")
        return final_message

    def handle_tool_call(self, tool_call):
        """
        Execute the tool specified in the tool call.
        Implements tools: 'echo' and 'reverse'.
        
        Args:
            tool_call (dict): Contains 'id', 'type', and 'function' (with 'name' and 'arguments').
        
        Returns:
            str: The output from executing the tool.
        """        
        function_data = tool_call.get("function", {})
        name = function_data.get("name")
        
        # Parse arguments if provided; they are passed as a JSON string by the API
        import json
        try:
            args = json.loads(function_data.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}

        tool = self.tool_registry.get_tool(name)
        if tool:
            result = tool.function(**args)
            return result

        return f"[Tool '{name}' not found]"
