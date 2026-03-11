import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TypeAlias

import instructor
import numpy as np
import requests
from openai import OpenAI
from pydantic import BaseModel
from prompt_spec import PromptTemplate

from .client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)


CHAT_MESSAGE_DICT: TypeAlias = Dict[str, str]


@dataclass
class InstructorClient(LLMClient):
    """
    LLMClient implementation backed by pydantic-instructor.

    This client extends the base LLMClient to support structured outputs
    via the `instructor` library.

    Parameters
    ----------
    instructor_mode : instructor.Mode
        The mode for the instructor client (e.g., JSON, TOOLS). Default is JSON.

    Attributes
    ----------
    _client : Any
        The initialized instructor client wrapper.
    """

    instructor_mode: instructor.Mode = instructor.Mode.JSON
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize the Instructor wrapper around the OpenAI client.
        """
        super().__post_init__()
        self._client = instructor.from_openai(
            self.base_client,
            mode=self.instructor_mode,
        )

    def complete[T: BaseModel](
        self,
        messages: List[CHAT_MESSAGE_DICT],
        response_model: Optional[type[T]] = None,
        show_prompt: bool = False,
        **kwargs: Any,
    ) -> Union[str, T]:
        """
        Run a chat completion.

        If `response_model` is provided, structured output is returned based
        on the Pydantic model. Otherwise, plain text is returned.

        Parameters
        ----------
        messages : list of dict
            The list of chat messages (e.g., `[{'role': 'user', 'content': '...'}]`).
        response_model : type[T], optional
            A Pydantic model class (T) to structure the response.
            Must be a subclass of BaseModel.
        show_prompt : bool, optional
            If True, logs the rendered prompt before sending. Default is False.
        **kwargs : Any
            Additional arguments passed to `chat.completions.create`.

        Returns
        -------
        Union[str, T]
            The response string (if no model provided) or an instance of T (the Pydantic model).

        Raises
        ------
        LLMClientError
            If the completion request fails.
        """
        if show_prompt:
            # Note: render_prompt_messages only uses the 'messages' list.
            rendered_text = self.render_prompt_messages(messages=messages)
            logger.info(f"SENDING PROMPT:\n{rendered_text}")

        try:
            result = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_model=response_model,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Instructor completion failed: {e}")
            raise LLMClientError("Instructor completion failed") from e

        if response_model is not None:
            # result is already an instance of T here
            return result

        return result.choices[0].message.content

    # TODO: The next two messages should be in prompt_spec!
    def messages_from_prompt_template(
        self, prompt_template: Optional[PromptTemplate], text: str
    ) -> List[Dict[str, str]]:
        """
        Generate a list of messages from a PromptTemplate.

        Raises
        ------
        NotImplementedError
            This method is currently dropped for Template in LinkML.
        """
        raise NotImplementedError("Method dropped for Template in LinkML")
        
        # Unreachable code preserved for reference/future implementation
        # messages = [{"role": "system", "content": prompt_template.system}] if self.system_message else []
        # for example in prompt_template.examples:
        #     messages.append({"role": "user", "content": example.input})
        #     messages.append({"role": "assistant", "content": example.output.model_dump_json(indent=2)})
        # messages.append({"role": "user", "content": text})
        # return messages

    def render_prompt_messages(self, messages: List[CHAT_MESSAGE_DICT]) -> str:
        """
        Render a list of chat messages into a single string for logging or display.

        Parameters
        ----------
        messages : list of dict
            The chat history.

        Returns
        -------
        str
            A formatted string representation of the conversation.
        """
        lines = []
        for msg in messages:
            role_label = "System:"
            if msg['role'] == "user":
                role_label = "Input:"
            elif msg['role'] == "assistant":
                role_label = "Output:"
            
            lines.append(f"{role_label} {msg['content']}")
            
        return "\n".join(lines)
