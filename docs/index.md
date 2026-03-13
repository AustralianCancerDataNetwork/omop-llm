# OMOP LLM Interface

`omop-llm` is designed as a simple LLM interface in the context of the OMOP CDM. Particularly, the wrapper currently exposes two interfaces:

!!! warning

    The backend is realised as an OpenAI client that would support a wide variety of models but we currently only support Ollama. Extension for this is planned in future releases.

- **`LLMClient`**: Base client with the capacity to:
    - obtain metadata information
    - calculate embeddings on demand
    - calculate semantic similarity
- **`InstructorClient`**: Child client of `LLMClient` to:
    - provide an interface for the [`instructor`](https://python.useinstructor.com/) library with easy instantiation.
    - chat completions using chat messages.


## Documentation overview
- [Installation](usage/installation.md)
