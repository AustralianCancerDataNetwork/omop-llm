import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TypeAlias

import instructor
import numpy as np
import requests
from openai import OpenAI
from pydantic import BaseModel
from prompt_spec import PromptTemplate

logger = logging.getLogger(__name__)


CHAT_MESSAGE_DICT: TypeAlias = Dict[str, str]


class LLMClientError(RuntimeError):
    """
    Custom exception for LLM Client runtime errors.
    """
    pass


@dataclass
class LLMClient:
    """
    Base class for LLM clients.

    This class replicates the API of the OntoGPT LLMClient but serves as a
    base for other implementations (e.g., InstructorClient). Relies on the
    OpenAI client for core functionality.

    Parameters
    ----------
    model : str
        The name of the model to use (e.g., 'gpt-4', 'llama3').
    api_base : str
        The base URL for the API endpoint.
    api_key : str
        The API key for authentication.
    temperature : float, optional
        The temperature parameter for generation. Default is 1.0.
    system_message : str, optional
        The default system message to prepend to chats. Default is "".

    Attributes
    ----------
    _base_client : OpenAI
        The initialized OpenAI client instance.
    _embedding_dim : int or None
        Cached embedding dimension size.
    """

    model: str
    api_base: str
    api_key: str = "ollama"  # required by OpenAI client, ignored by Ollama
    temperature: float = 1.0
    system_message: str = ""
    embedding_batch_size: int = 32
    _base_client: OpenAI = field(init=False, repr=False)
    _embedding_dim: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        logger.info(f"Initialising {self.__class__.__name__} for model={self.model}")
        if self.api_key is None:
            self.api_key = "ollama"  # Default to "ollama" for compatibility, but can be overridden

        self._base_client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    @property
    def embedding_dim(self) -> int:
        """
        Retrieve the embedding dimension for the current model.

        If the dimension is not cached, it attempts to fetch it from the API.
        Currently supports Ollama endpoints.

        Returns
        -------
        int
            The size of the embedding vector.

        Raises
        ------
        ValueError
            If model information cannot be found in the Ollama response.
        NotImplementedError
            If the API base is not supported for automatic dimension retrieval.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        
        if (
            "ollama" in self.api_base or 
            (
                (
                    "localhost" in self.api_base or
                    "127.0.0.1" in self.api_base
                ) and self.api_key == "ollama"
            )
        ):
            # Strip /v1 to access base Ollama API
            ollama_url_without_v1 = self.api_base.replace("/v1", "")
            requests_url = f"{ollama_url_without_v1}/api/show"
            
            response = requests.post(requests_url, json={"name": self.model}).json()
            model_info = response.get("model_info", {})
            
            if model_info:
                # Find keys resembling 'embedding_length'
                embedding_key = [key for key in model_info.keys() if "embedding_length" in key]
                if len(embedding_key) == 1:
                    self._embedding_dim = int(model_info[embedding_key[0]])
                    return self._embedding_dim
            
            raise ValueError(f"Model information not found in Ollama response: {response}")
        else:
            raise NotImplementedError("Embedding dimension retrieval not implemented for this API base")

    @property
    def base_client(self) -> OpenAI:
        return self._base_client

    def embeddings(self, text: Union[str, List[str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Retrieve embeddings for the given text.

        Parameters
        ----------
        text : str or List[str]
            The input text or list of texts to embed.
        batch_size : int, optional
            The number of texts to process in a single API call. Default is 32.

        Returns
        -------
        np.ndarray
            A 2D numpy array containing the embeddings.

        Raises
        ------
        AssertionError
            If the base client has not been initialized.
        """
        assert self.base_client is not None, "Base client should be initialized"
        if batch_size is None:
            batch_size = self.embedding_batch_size
        
        if isinstance(text, str):
            text = [text]

        batch_buffer = []

        for batch_chunk_idx in range(0, len(text), batch_size):
            logger.debug(f"Processing batch chunk from index {batch_chunk_idx} to {batch_chunk_idx + batch_size}")
            batch_chunk = text[batch_chunk_idx:batch_chunk_idx + batch_size]
            response = self.base_client.embeddings.create(
                model=self.model,
                input=batch_chunk,
            )
            batch_buffer.extend([emb.embedding for emb in response.data])

        return np.array(batch_buffer)

    def similarity(
        self,
        terms: Union[str, List[str], np.ndarray],
        terms_to_match: Union[str, List[str], np.ndarray],
        **kwargs: Any
    ) -> np.ndarray:
        """
        Calculate the cosine similarity between two sets of terms.

        This method handles inputs as strings, lists of strings, or pre-computed
        numpy arrays of embeddings.

        Parameters
        ----------
        terms : str, List[str], or np.ndarray
            The source terms or embeddings.
        terms_to_match : str, List[str], or np.ndarray
            The target terms or embeddings to match against.
        **kwargs : Any
            Additional arguments passed to the embedding function if embedding is required.

        Returns
        -------
        np.ndarray
            A similarity matrix.

        Raises
        ------
        ValueError
            If inputs are not strings, lists, or numpy arrays.
        """
        if isinstance(terms, str):
            terms = [terms]
        if isinstance(terms_to_match, str):
            terms_to_match = [terms_to_match]

        # Process source terms
        if isinstance(terms, list):
            terms_embeddings = self.embeddings(text=terms, **kwargs)
        elif isinstance(terms, np.ndarray):
            terms_embeddings = terms
        else:
            raise ValueError("terms must be either a string, list of strings, or numpy array")

        # Process target terms
        if isinstance(terms_to_match, list):
            terms_to_match_embeddings = self.embeddings(text=terms_to_match, **kwargs)
        elif isinstance(terms_to_match, np.ndarray):
            terms_to_match_embeddings = terms_to_match
        else:
            raise ValueError("terms_to_match must be either a string, list of strings, or numpy array")

        return self.cosine_similarity(terms_embeddings, terms_to_match_embeddings)

    @staticmethod
    def cosine_similarity(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        """
        Compute the cosine similarity between two matrices of vectors.

        Parameters
        ----------
        vecs_a : np.ndarray
            A 2D array of vectors (Shape: M x D).
        vecs_b : np.ndarray
            A 2D array of vectors (Shape: N x D).

        Returns
        -------
        np.ndarray
            The dot product of the normalized vectors (Shape: M x N).

        Notes
        -----
        A small epsilon (1e-10) is added to the norms to prevent division by zero.
        """
        assert vecs_a.ndim == 2 and vecs_b.ndim == 2, "Input vectors must be 2D arrays"
        
        norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)

        # Prevent division by zero
        norm_a[norm_a == 0] = 1e-10
        norm_b[norm_b == 0] = 1e-10

        vecs_a_norm = vecs_a / norm_a
        vecs_b_norm = vecs_b / norm_b

        return np.dot(vecs_a_norm, vecs_b_norm.T)

    def euclidean_distance(self, text1: str, text2: str, **kwargs: Any) -> float:
        """
        Calculate the Euclidean distance between embeddings of two texts.

        Parameters
        ----------
        text1 : str
            The first text string.
        text2 : str
            The second text string.
        **kwargs : Any
            Additional arguments passed to the embedding function.

        Returns
        -------
        float
            The Euclidean distance (L2 norm) between the two embedding vectors.
        """
        a1 = self.embeddings(text1, **kwargs)
        a2 = self.embeddings(text2, **kwargs)
        return float(np.linalg.norm(np.array(a1) - np.array(a2)))