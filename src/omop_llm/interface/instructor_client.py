"""Instructor-backed client for structured LLM completions.

Completely separate from :class:`~omop_llm.interface.client.EmbeddingClient`.
They may point at the same ``api_base`` (typical for Ollama setups) but serve
different purposes and carry different parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import instructor
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from .client import CHAT_MESSAGE_DICT, LLMClientError

logger = logging.getLogger(__name__)


class InstructorClient:
    """Client for structured LLM completions backed by instructor.

    Wraps an OpenAI-compatible transport with instructor's pydantic-validation
    layer.  Pass a ``response_model`` to :meth:`complete` to get a validated
    pydantic instance; omit it for plain-text output.

    Parameters
    ----------
    model : str
        Model name passed verbatim to the chat completions API.
    api_base : str
        API endpoint base URL, e.g. ``'http://localhost:11434/v1'``.
    api_key : str, optional
        API key.  Defaults to ``'ollama'``.
    temperature : float, optional
        Generation temperature.  Default is 1.0.
    instructor_mode : instructor.Mode, optional
        Instructor response mode.  Default is ``JSON``.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "ollama",
        temperature: float = 1.0,
        instructor_mode: instructor.Mode = instructor.Mode.JSON,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._instructor_mode = instructor_mode
        base_client = OpenAI(base_url=api_base, api_key=api_key)
        self._instructor_client = instructor.from_openai(base_client, mode=instructor_mode)  # type: ignore[arg-type]
        logger.info(f"InstructorClient initialised for model={self._model!r}")

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def instructor_client(self):
        """The underlying instructor-patched client instance."""
        return self._instructor_client

    def complete(
        self,
        messages: List[ChatCompletionMessageParam],
        response_model: type[BaseModel],
        show_prompt: bool = False,
        **kwargs: Any,
    ) -> BaseModel:
        """Run a chat completion.

        Parameters
        ----------
        messages : list of dict
            Chat messages, e.g. ``[{'role': 'user', 'content': '...'}]``.
        response_model : type[T], optional
            Pydantic model for structured output.  When provided, returns a
            validated instance of *T*.
        show_prompt : bool, optional
            Log the rendered prompt before sending.  Default is ``False``.
        **kwargs
            Forwarded to ``chat.completions.create``.

        Returns
        -------
        str | T
            Plain assistant text, or a validated pydantic instance.

        Raises
        ------
        LLMClientError
            If the completion request fails.
        """
        if show_prompt:
            logger.info(f"SENDING PROMPT:\n{self.render_prompt_messages(messages)}")

        try:
            # instructor overloads create() at runtime; the overload that returns T
            # when response_model is set is not visible to static analysers.
            result = self._instructor_client.chat.completions.create(  # type: ignore[call-overload]
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                response_model=response_model,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Instructor completion failed: {e}")
            raise LLMClientError("Instructor completion failed") from e

        if response_model is not None:
            return result
        else:
            raise RuntimeError(
                "Received a response model from instructor, but response_model was None. "
                "This should never happen. Please report this issue."
            )

    def render_prompt_messages(self, messages: List[ChatCompletionMessageParam]) -> str:
        """Render a message list to a single string for logging."""
        return "\n".join(
            f"{msg['role']}: {msg.get('content', '')}"
            for msg in messages
        )
