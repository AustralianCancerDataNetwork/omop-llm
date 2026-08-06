# omop-llm

Shared chat/embedding backend contract for the OMOP stack, built on [any-llm](https://github.com/mozilla-ai/any-llm). One typed `ModelBackend` interface (sync and async methods, both real), a closed set of supported providers (local: `ollama`, `llamacpp`, `vllm`; cloud: `openai`, `anthropic`, `gemini`), and explicit capability declarations instead of provider-name guessing. Extended documentation can be found [here](https://AustralianCancerDataNetwork.github.io/omop-llm).

```python
from omop_llm import build_model_backend

backend = build_model_backend(provider="ollama", model="llama3.2:8b", base_url="http://localhost:11434")

# async
response = await backend.async_complete([{"role": "user", "content": "Hello"}])

# sync
response = backend.complete([{"role": "user", "content": "Hello"}])
```

Or resolved from an `oa-configurator` stack config:

```python
from oa_configurator import Resolver, load_stack_config
from omop_llm import build_model_backend_from_resolved

resolved = Resolver(load_stack_config()).resolve_model("embed-default")
backend = build_model_backend_from_resolved(resolved)
```

See [docs/index.md](docs/index.md) for the full design (provider registry, capability model, structured extraction).
