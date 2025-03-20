"""
DeepSeek Agent implementation.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union, Generator

from moya.agents.base_agent import Agent, AgentConfig  # Changed BaseAgent to Agent


class DeepSeekAgent(Agent):  # Also changed BaseAgent to Agent here
    """
    An agent that uses DeepSeek's API for text generation.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.config = config  # Store config explicitly
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", config.llm_config.get("api_key", ""))
        self.model_name = config.llm_config.get("model_name", "deepseek-coder")
        self.temperature = config.llm_config.get("temperature", 0.7)
        self.base_url = config.llm_config.get("base_url", "https://api.deepseek.com/v1")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare HTTP headers for API calls."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _prepare_messages(self, user_input: str) -> List[Dict[str, str]]:
        """Convert user input to message format expected by DeepSeek API."""
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def handle_message(self, message: str) -> str:
        """
        Process a message and return a response using the DeepSeek API.
        
        Args:
            message: The user's message
            
        Returns:
            str: The model's response
        """
        try:
            messages = self._prepare_messages(message)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 2000,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._prepare_headers(),
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "[DeepSeekAgent error: No response content]"
                
        except Exception as e:
            return f"[DeepSeekAgent error: {str(e)}]"

    def handle_message_stream(self, message: str) -> Generator[str, None, None]:
        """
        Process a message and stream the response using the DeepSeek API.
        
        Args:
            message: The user's message
            
        Yields:
            str: Chunks of the model's response
        """
        try:
            messages = self._prepare_messages(message)
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 2000,
                "stream": True
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._prepare_headers(),
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data:"):
                        data = line[5:].strip()  # Changed from trip() to strip()
                        if data != "[DONE]":
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                    
        except Exception as e:
            yield f"[DeepSeekAgent error: {str(e)}]"