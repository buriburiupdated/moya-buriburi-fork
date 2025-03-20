import sys
import os
from typing import Dict, Any
from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
import re

class TextAutocomplete:
    """Tools for text autocompletion capabilities."""

    _agent = None
    _last_input = None
    _last_output = None

    def _get_agent(self):
        if TextAutocomplete._agent is None:
            agent_config = AzureOpenAIAgentConfig(
                agent_name="text_completer",
                agent_type="CompletionAgent",
                model_name="gpt-4o",
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
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Use default OpenAI API base
                api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
                organization=None  # Use default organization
            )
            TextAutocomplete._agent = AzureOpenAIAgent(agent_config)
            test_response = TextAutocomplete._agent.handle_message("test")
            if not test_response:
                raise Exception("No response from Azure OpenAI")
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

            # Extract the first 5-6 words from the response
            completion = " ".join(response.split()[:6])
            completion = self._clean_completion(completion)

            TextAutocomplete._last_input = text
            TextAutocomplete._last_output = completion

            return completion

        except Exception as e:
            print(f"\nError getting completion: {e}")
            return ""

    def configure_autocomplete_tools(self) -> None:
        pass