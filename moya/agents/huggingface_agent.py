"""
HuggingFaceAgent for Moya.

An Agent that uses HuggingFace's models and Inference API
to generate responses, supporting both local models and the HF API.
"""

import os
import json
import time
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Iterator, Union

from moya.agents.base_agent import Agent
from moya.agents.base_agent import AgentConfig
from moya.tools.base_tool import BaseTool
from moya.tools.tool_registry import ToolRegistry
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
from threading import Thread


@dataclass
class HuggingFaceAgentConfig(AgentConfig):
    """
    Configuration data for a HuggingFaceAgent.
    """
    # Use field() with default_factory to mark fields as required
    model_name: str = field(default=None)  # Required but doesn't violate dataclass rules
    task: str = field(default=None)        # Required but doesn't violate dataclass rules
    access_token: Optional[str] = None
    use_api: bool = False
    api_url: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    device: str = "cpu"
    quantization: Optional[str] = None
    
    def __post_init__(self):
        """Validate required fields after initialization"""
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        
        if not self.model_name:
            raise ValueError("model_name is required for HuggingFaceAgentConfig")
        if not self.task:
            raise ValueError("task is required for HuggingFaceAgentConfig")


class HuggingFaceAgent(Agent):
    """
    A versatile HuggingFace-based agent that can use any model from the HF Hub.
    
    Supports:
    - Local models and API-based models
    - Various tasks (text generation, Q&A, translation, etc.)
    - Streaming responses
    - Tool usage for capable models
    - Quantization for efficient local inference
    """
    
    def __init__(self, config: HuggingFaceAgentConfig):
        """Initialize the HuggingFaceAgent."""
        super().__init__(config=config)
        
        # Store configuration
        self.model_name = config.model_name
        self.task = config.task
        self.access_token = config.access_token or os.environ.get("HF_TOKEN")
        self.system_prompt = config.system_prompt
        self.use_api = config.use_api
        self.api_url = config.api_url
        self.generation_config = config.generation_config or {}
        self.device = config.device
        self.quantization = config.quantization
        
        # Placeholder for pipeline
        self.pipeline = None
        self.tokenizer = None
        
        # Configure pipeline based on task type
        self._configure_pipeline()
        
        # Additional configuration for response parsing
        self._configure_response_handling()

    def _configure_pipeline(self):
        """Configure the appropriate pipeline based on configuration."""
        # Set default parameters for the pipeline
        pipeline_kwargs = {
            'task': self.task,
            'model': self.model_name,
        }
        
        if self.access_token:
            pipeline_kwargs['token'] = self.access_token
            
        if not self.use_api:
            # Local model configuration
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                import torch
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                pipeline_kwargs['quantization_config'] = bnb_config
                
            elif self.quantization == "8bit":
                from transformers import BitsAndBytesConfig
                import torch
                
                # Configure 8-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                )
                pipeline_kwargs['quantization_config'] = bnb_config
            
            # Set device for local models
            pipeline_kwargs['device'] = self.device
            
            # Use AutoTokenizer for better compatibility
            pipeline_kwargs['tokenizer'] = AutoTokenizer.from_pretrained(self.model_name)
        else:
            # API-based configuration
            if self.api_url:
                pipeline_kwargs['api_url'] = self.api_url
        
        # Create the pipeline
        self.pipeline = pipeline(**pipeline_kwargs)
        
        # Store the tokenizer for streaming
        if hasattr(self.pipeline, "tokenizer"):
            self.tokenizer = self.pipeline.tokenizer
        elif 'tokenizer' in pipeline_kwargs:
            self.tokenizer = pipeline_kwargs['tokenizer']
        else:
            # Fallback to loading tokenizer explicitly
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.access_token)

    def _configure_response_handling(self):
        """Configure response handling based on the task type."""
        # For chat models with specific formatting
        if 'llama' in self.model_name.lower() or 'mistral' in self.model_name.lower():
            self.chat_template = True
        else:
            self.chat_template = False
            
        # For models that need special prompt formatting
        if self.task in ['text-generation', 'text2text-generation']:
            self.format_chat = True
        else:
            self.format_chat = False

    def handle_message(self, message: str, **kwargs) -> str:
        """Handle a user message and return a response."""
        thread_id = kwargs.get("thread_id", "default")
        
        # Format the prompt with the system prompt if available
        prompt = self._format_prompt(message, thread_id)
        
        # Get the model's generation config
        generation_config = self._get_generation_config(kwargs)
        
        # Call the model
        try:
            response = self.pipeline(prompt, **generation_config)
            
            # Process the response based on the task type
            processed_response = self._process_response(response)
            
            # Check for and handle tool calls if the model supports them
            processed_response = self._handle_tools_if_needed(processed_response, thread_id)
            
            return processed_response
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def _clean_streaming_chunk(self, chunk: str) -> str:
        """Clean a streaming chunk of any special tokens or formatting."""
        # Remove model-specific tags
        chunk = re.sub(r'<\|assistant\|>', '', chunk)
        chunk = re.sub(r'</s>', '', chunk)
        
        # Remove "Bot:" or "Assistant:" prefix if it's at the beginning
        chunk = re.sub(r'^(Bot|Assistant):\s*', '', chunk)
        
        return chunk

    def handle_message_stream(self, message: str, **kwargs) -> Iterator[str]:
        """Handle a user message and yield a streaming response."""
        thread_id = kwargs.get("thread_id", "default")
        
        # Format the prompt with the system prompt if available
        prompt = self._format_prompt(message, thread_id)
        
        # Get the model's generation config
        generation_config = self._get_generation_config(kwargs)
        
        # Set up the streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, timeout=10.0
        )
        
        # Function to run the model in a separate thread
        def run_model():
            try:
                self.pipeline(
                    prompt, 
                    streamer=streamer,
                    **generation_config
                )
            except Exception as e:
                streamer.add_token(f"Error: {str(e)}")
                streamer.end()
        
        # Start the model in a separate thread
        Thread(target=run_model).start()
        
        # For cleaning up streaming output
        response_buffer = ""
        current_chunk = ""
        stop_streaming = False
        
        # Yield tokens as they become available
        for token in streamer:
            if stop_streaming:
                break
                
            current_chunk += token
            response_buffer += token
            
            # Check for stopping conditions
            if "User:" in current_chunk or "Human:" in current_chunk or "</s>" in current_chunk:
                stop_streaming = True
                # Clean up the response
                cleaned_chunk = self._clean_streaming_chunk(current_chunk)
                if cleaned_chunk:
                    yield cleaned_chunk
                break
            
            # Reset current chunk after yielding to avoid checking the same text multiple times
            if len(current_chunk) > 20:
                cleaned_chunk = self._clean_streaming_chunk(current_chunk)
                if cleaned_chunk:
                    yield cleaned_chunk
                current_chunk = ""

    def _format_prompt(self, message: str, thread_id: str) -> str:
        """Format the prompt based on the model and task type."""
        if not self.format_chat:
            return message
            
        # Get conversation history if memory tools are configured
        history = ""
        if hasattr(self, "memory_tool") and thread_id:
            # This would retrieve conversation history if implemented
            pass
            
        # For models with chat templates
        if self.chat_template:
            messages = []
            
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
                
            if history:
                # Add history messages if available
                pass
                
            messages.append({"role": "user", "content": message})
            
            # Use the tokenizer's chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Default formatting for other models
        prompt = ""
        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n\n"
        
        if history:
            prompt += f"{history}\n\n"
            
        prompt += f"User: {message}\n\nAssistant: "
        
        return prompt

    def _get_generation_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get generation configuration, merging defaults with any overrides."""
        config = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.95,
            'do_sample': True,
        }
        
        # Add the model's default generation config
        config.update(self.generation_config or {})
        
        # Add any overrides from kwargs
        if 'generation_config' in kwargs:
            config.update(kwargs['generation_config'])
            
        return config

    def _process_response(self, response: Any) -> str:
        """Process the raw model response into a clean text response."""
        if isinstance(response, list) and response:
            # Most HF pipelines return a list of dicts
            if isinstance(response[0], dict):
                if 'generated_text' in response[0]:
                    text = response[0]['generated_text']
                elif 'summary_text' in response[0]:
                    text = response[0]['summary_text']
                elif 'answer' in response[0]:
                    text = response[0]['answer']
                else:
                    # Fallback to string representation if structure is unknown
                    text = str(response[0])
            else:
                # If it's just a list of strings or other objects
                text = str(response[0])
        elif isinstance(response, str):
            # Some pipelines directly return strings
            text = response
        else:
            # Fallback for other return types
            text = str(response)
            
        # Clean up the response
        return self._clean_response(text)

    def _clean_response(self, text: str) -> str:
        """Clean the response by removing model-specific tags and preventing self-chat."""
        # Remove model-specific tags
        text = re.sub(r'<\|assistant\|>', '', text)
        text = re.sub(r'</s>', '', text)
        
        # Remove any "Bot:" or "Assistant:" prefixes
        text = re.sub(r'^(Bot|Assistant):\s*', '', text)
        
        # Prevent self-chat by stopping at any "User:" or new speaker indicator
        if "User:" in text:
            text = text.split("User:")[0]
        if "Human:" in text:
            text = text.split("Human:")[0]
            
        # Remove the prompt from the response
        prompt_pattern = r"User:.*?Assistant:"
        text = re.sub(prompt_pattern, "", text, flags=re.DOTALL)
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        return text

    def _model_supports_tools(self) -> bool:
        """Determine if the model is likely to support tool use."""
        # Check based on model name or configuration
        tool_capable_models = [
            "llama", "mistral", "phi", "gemma", "mpt", 
            "falcon", "gpt-"
        ]
        
        return any(model in self.model_name.lower() for model in tool_capable_models)

    def _handle_tools_if_needed(self, response: str, thread_id: str) -> str:
        """Check for tool calls in the response and handle them if present."""
        if not self.tool_registry or not self._model_supports_tools():
            return response
            
        # Look for tool calls in the response
        # This pattern will need to be adjusted based on model output format
        tool_pattern = r"```json\s*({[^`]*})\s*```"
        matches = re.findall(tool_pattern, response)
        
        if not matches:
            return response
            
        # Process each tool call
        for match in matches:
            try:
                tool_call = json.loads(match)
                result = self._process_tool_call(tool_call)
                
                # Replace the tool call with the result
                response = response.replace(f"```json\n{match}\n```", f"[Tool result: {result}]")
            except json.JSONDecodeError:
                pass
                
        return response

    def _process_tool_call(self, tool_call_text: str) -> str:
        """Process a tool call and return the result."""
        # Clean up the text to extract JSON
        json_text = re.sub(r"```json|```", "", tool_call_text).strip()
        
        try:
            tool_data = json.loads(json_text)
            
            # Extract tool name and arguments
            name = tool_data.get("name", tool_data.get("tool_name", None))
            args = tool_data.get("arguments", tool_data.get("parameters", {}))
            
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a single argument
                    args = {"input": args}
            
            # Get the tool and call it
            tool = self.tool_registry.get_tool(name)
            if tool:
                return tool.function(**args)
            else:
                return f"[Tool '{name}' not found]"
        except Exception as e:
            return f"[Error processing tool call: {str(e)}]"