# Providers

`omop-llm` intentionally supports a select collection of `any-llm`'s own providers in two categories: 
- local (`ollama`, `llamacpp`, `vllm`), and 
- cloud (`openai`, `anthropic`, `gemini`). 

See any-llm's own [provider reference](https://docs.mozilla.ai/any-llm/providers/) for capabilities of these providers. 
Given the interface we have devised, future providers will be extended in the future to support other use-cases.

## `base_url` and `api_key`

Every one of these fields is optional on `build_backend(provider, model, base_url=None, api_key=None, ...)`. What "not set" resolves to differs per provider.
Resolution order for both, always: **explicit argument → the provider's own environment variable → a class-level default (if any)**.

| Provider | Default `base_url` when not set | `api_key` required? |
|---|---|---|
| `ollama` | any-llm sets none; falls through to the official `ollama` SDK's own default (`http://localhost:11434`) | No |
| `llamacpp` | `http://127.0.0.1:8080/v1` (any-llm's own default, matches `llama-server`'s conventional port) | No |
| `vllm` | `http://localhost:8000/v1` (any-llm's own default, matches vLLM's conventional port) | No (any-llm's `VllmProvider` explicitly overrides key verification, since self-hosted vLLM commonly runs without auth) |
| `openai` | `https://api.openai.com/v1` (any-llm's own explicit default) | Yes |
| `anthropic` | any-llm sets none; falls through to the `anthropic` SDK's own default (the real Anthropic API) | Yes |
| `gemini` | any-llm sets none; falls through to the `google-genai` SDK's own default (the real Gemini API) | Yes |

! note "The pattern"
Local providers either have no sensible universal default or a conventional local-dev default. You'll almost always want to set `base_url` explicitly once you're pointed at anything other than a single local instance on the default port. Cloud providers need no `base_url` at all for the normal case: leaving it unset resolves to the real vendor API, exactly as if you were calling that vendor's own SDK directly with no `base_url` override. You only set `base_url` for a cloud provider to point at something *other* than the vendor's real endpoint (an Azure OpenAI-style proxy, for instance).

| Provider | Env var for `base_url` | Env var for `api_key` |
|---|---|---|
| `ollama` | `OLLAMA_HOST` | none (not required) |
| `llamacpp` | `LLAMACPP_API_BASE` | none (not required) |
| `vllm` | `VLLM_API_BASE` | `VLLM_API_KEY` (optional) |
| `openai` | `OPENAI_BASE_URL` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_BASE_URL` | `ANTHROPIC_API_KEY` |
| `gemini` | `GOOGLE_GEMINI_BASE_URL` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

## Capabilities per provider

| Provider | `streaming` | `embeddings` | `extended_thinking` | `tool_use` | `structured_output` | Notes |
|---|---|---|---|---|---|---|
| `ollama` | ✅ | ✅ | ✅ | ✅ | ✅ | Canonical model names require an explicit tag (`llama3:8b`, not `llama3` or `llama3:latest`); see `omop_llm.providers.supported.OllamaProvider.canonical_model_name`. |
| `llamacpp` | ✅ | ✅ | ✅ | ✅ | ✅ | Covers both a local `llama-server` and a CUDA/TRE fallback profile; only `base_url` changes. |
| `vllm` | ✅ | ✅ | ✅ | ✅ | ✅ | Preferred TRE/NVIDIA backend. |
| `openai` | ✅ | ✅ | ❌ | ✅ | ✅ | |
| `anthropic` | ✅ | ❌ | ✅ | ✅ | ✅ | No embeddings API; `ModelBackend.embed_texts`/`async_embed_texts` refuse this provider. |
| `gemini` | ✅ | ✅ | ✅ | ✅ | ✅ | |

`streaming`/`embeddings`/`extended_thinking` come from any-llm's own `get_provider_metadata()`, verified directly against the installed package for these six providers. `tool_use`/`structured_output` are declared by `omop-llm` itself, since any-llm tracks neither (see [`ModelCapabilities`](reference.md#omop_llm.capabilities)).

## Adding a provider

- Add a class to `omop_llm.providers.supported`, subclassing both `ProviderMixin` and any-llm's own provider class for it (imported under an `AnyLLM`-prefixed alias to avoid a name collision with the new class). 
- `canonical_model_name` is a required override, not a default passthrough, on purpose: it forces a conscious decision about that provider's naming rules rather than silently inheriting "no transformation needed." 
- `PROVIDER_REGISTRY` (`omop_llm.providers.registry`) picks the new class up automatically, no separate list to update.
