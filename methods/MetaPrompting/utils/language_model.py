# Import the necessary libraries
import os
import openai
from openai import OpenAI
from typing import Any, Dict, List, Optional, Union
import retry


class LanguageModel:
    """Abstract class for a language model."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.stop_tokens = stop_tokens or []
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        raise NotImplementedError("generate() not implemented.")

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_name

    def get_stop_tokens(self) -> List[str]:
        """Get the stop tokens."""
        return self.stop_tokens

    def get_kwargs(self) -> Any:
        """Get the kwargs."""
        return self.kwargs

    def set_kwargs(self, kwargs: Any) -> None:
        """Set the kwargs."""
        self.kwargs = kwargs


class DummyLanguageModel(LanguageModel):
    """A dummy language model that just returns the prompt."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, stop_tokens, **kwargs)

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        return [prompt] * num_return_sequences


class OpenAI_LanguageModel(LanguageModel):
    """A language model from OpenAI's API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_type: str= None,
        api_version: str= None,
        api_base: str= None,
        stop_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the OpenAI API.

        Args:
            model_name (str): The name of the model to use.
            api_key (str): The API key to use.
            api_type (str): The API type to use.
            api_version (str): The API version to use.
            api_base (str): The API base to use.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            None
        """
        # Set the OpenAI API parameters
        self.model_name = model_name
        self.api_key = api_key
        self.api_type = api_type
        self.api_version = api_version
        self.api_base = api_base
        self.stop_tokens = stop_tokens

        # Set the model type
        self.model_type = "chat"

        self.num_llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._images = None

        self.client = None

        ## OLD AZURE API CODE
        if self.api_type == "azure":
            openai.api_key = self.api_key
            openai.api_type = self.api_type
            openai.api_version = self.api_version
            openai.api_base = self.api_base

            if self.model_name in [
                "code-davinci-002",
                "text-davinci-002",
                "text-davinci-003",
            ]:
                self.model_type = "completion"
            elif self.model_name in [
                "gpt-4",
                "gpt-4-32k",
                "gpt-35-turbo",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-35-turbo-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                "gpt-35-turbo-0613",
                "gpt-35-turbo",
            ]:
                self.model_type = "chat"
            else:
                raise ValueError(f"Model {self.model_name} not supported.")
        else:
            ## NEW OPENAI API CODE SUPPORT
            # Skipping model name validation for now
            # Set up the client
            api_key = os.environ.get("OPENAI_API_KEY") or self.api_key
            base_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
                or self.api_base
            )
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def set_images(self, img_paths):
        self._images = img_paths

    def _inject_images(self, messages):
        from methods.mas_base import _encode_media_to_content_parts
        msgs = [m.copy() for m in messages]
        for msg in msgs:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                parts = []
                for img in self._images:
                    parts.extend(_encode_media_to_content_parts(img))
                parts.append({"type": "text", "text": msg["content"]})
                msg["content"] = parts
                break
        return msgs

    @retry.retry(tries=3, delay=1)
    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate text based on a prompt or messages.

        Args:
            prompt_or_messages (Union[str, List[Dict[str, str]]]): The prompt or messages to generate text from.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.
            temperature (float, optional): The temperature to use. Defaults to 0.7.
            top_p (float, optional): The top p to use. Defaults to 1.0.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[str]: The list of generated texts based on the prompt or messages.
        """
        # Set the stop tokens
        stop_tokens = stop_tokens or self.stop_tokens

        if self.api_type == "azure":
            ## OLD AZURE API CODE
            # Generate the text
            if self.model_type == "chat":
                response = openai.ChatCompletion.create(
                    engine=self.model_name,
                    messages=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of messages
                return [message["message"]["content"] for message in response.choices]
            else:
                response = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of outputs
                return [output["text"] for output in response.choices]
        else:
            if self._images and isinstance(prompt_or_messages, list):
                prompt_or_messages = self._inject_images(prompt_or_messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_or_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_return_sequences,
                stop=stop_tokens,
                **kwargs,
            )
            self.num_llm_calls += 1
            usage = getattr(response, "usage", None)
            if isinstance(usage, dict):
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
            elif usage is not None:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            return [output.message.content or "" for output in response.choices]
