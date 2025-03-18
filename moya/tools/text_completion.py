import sys
from typing import Dict, Any
from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
import re

class TextAutocomplete:
    """Tools for text autocompletion capabilities."""

    _agent = None
    _last_input = None
    _last_output = None

    def _get_agent(self):
        if TextAutocomplete._agent is None:
            agent_config = AgentConfig(
                agent_name="text_completer",
                agent_type="CompletionAgent",
                description="A human-like text completion assistant",
                system_prompt = """
                You are a text auto-complete bot designed to predict only the next **few words** in a given sentence.
                - Do **not** introduce yourself, greet the user, or answer questions.
                - Ignore text like "User:", "Bot:", or conversational cues such as "hello", "who", etc.
                - Focus solely on continuing the user's incomplete sentence with meaningful context.
                - Never start new ideas or provide explanations.
                - Your output should match the user's writing style and tone.
                - Responses must be **5-6 words** maximum and grammatically correct.
                """,
                llm_config={
                    'model_name': "mistral:latest",
                    'temperature': 0.3,  # Lowered for more controlled completions
                    'base_url': "http://localhost:11434",
                    'context_window': 1024
                }
            )
            TextAutocomplete._agent = OllamaAgent(agent_config)
            test_response = TextAutocomplete._agent.handle_message("test")
            if not test_response:
                raise Exception("No response from Ollama")
        return TextAutocomplete._agent

    def _clean_text(self, text: str) -> str:
        text = text.replace("User:", "").strip().rstrip(" .")
        return text

    def _clean_completion(self, completion: str) -> str:
        completion = completion.strip()
        completion = re.sub(r'\b\w{1,2}[.?!]*$', '', completion)  # Remove incomplete words
        return completion.strip()

    def complete_text(self, text: str) -> str:
        try:
            if text == TextAutocomplete._last_input:
                return TextAutocomplete._last_output

            agent = self._get_agent()
            text = self._clean_text(text)
            if not text:
                return ""

            response = agent.handle_message(text)
            if not response:
                return ""

            completion = " ".join(response.strip().split()[:6])  # Limit completion to 5-6 words
            completion = self._clean_completion(completion)

            TextAutocomplete._last_input = text
            TextAutocomplete._last_output = completion

            return completion

        except Exception as e:
            print(f"\nError getting completion: {e}")
            return ""

    def configure_autocomplete_tools(self) -> None:
        pass
