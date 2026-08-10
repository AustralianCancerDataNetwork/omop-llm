# OMOP LLM Interface

`omop-llm` is the shared chat/embedding backend contract for the OMOP stack: a generic interface for calling a chat or embedding model, so packages that need one compose it in rather than each writing their own adapter, capability model, and provider vocabulary from scratch.

## What it provides

- **`ModelBackend`** ([reference](reference.md#omop_llm.backend)): the one calling contract every consumer uses, built by `build_model_backend(provider, model, ...)`. It wraps a single [any-llm](https://github.com/mozilla-ai/any-llm) provider instance and exposes chat completion, embeddings, and structured extraction as methods on one object, each with a synchronous form and an `async_`-prefixed asynchronous form (`complete`/`async_complete`, `embed_texts`/`async_embed_texts`, and so on), so both async and fully synchronous consumers get a real, non-hand-rolled path.
- **Asymmetric embedding prefixes** ([`EmbeddingRole`](reference.md#omop_llm.embeddings), [full guide](usage/asymmetric-embeddings.md)): asymmetric embedding models (nomic-embed-text, the E5 family, BGE, and others) need a different prefix prepended depending on whether the text is being indexed or used to search. `embed_texts(texts, role=EmbeddingRole.DOCUMENT)`/`role=EmbeddingRole.QUERY` applies it automatically, sourced from `build_model_backend`'s own `document_prefix`/`query_prefix` arguments (or `oa-configurator`'s typed `ModelConfig.document_prefix`/`query_prefix` fields via `build_model_backend_from_resolved`) so the values live in one place, not duplicated per consuming package.
- **A closed provider registry** ([reference](reference.md#omop_llm.providers)): `omop-llm` supports selected providers, not any-llm's full set shown in the [Providers overview](providers.md). Every supported provider is a real subclass of any-llm's own provider class, which is both the allow-list (nothing outside this set is reachable through `omop-llm`) and the seam for provider-specific behavior, such as Ollama's canonical model naming and embedding-dimension fast path (see `omop_llm.providers.supported`).
- **An explicit, per-model capability model** ([`Capabilities`](reference.md#omop_llm.capabilities)): a resolved backend's effective capabilities are the provider's own ceiling (`provider_capabilities_for`, sourced from any-llm's metadata plus `omop-llm`'s own `tool_use`/`structured_output` declarations, since any-llm tracks neither) AND'd with what the *specific model* is declared to support (e.g. `oa-configurator`'s `ModelConfig.embeddings`/`tool_use`/`structured_output`/`extended_thinking`, since neither any-llm nor `omop-llm` can introspect this per model) — a capability is only effective if both agree. `streaming` is always `False`, since `ModelBackend` doesn't implement it regardless of what the provider/model support. A caller requiring a capability the resolved backend doesn't have fails at construction time, not mid-run.
- **Structured single-object extraction** (`ModelBackend.extract`/`async_extract`): pulling one validated Pydantic object out of one LLM call. This is *not* the same problem as multi-turn agentic tool use (a model calling several real tools across several turns), which stays on `ModelBackend.complete(messages, tools=...)` directly. See [`omop_llm.structured`](reference.md#omop_llm.structured)'s own docstring for why the primary strategy is any-llm's native `response_format=` translation, and why `instructor`-based extraction (the optional fallback) is only offered for `openai`/`llamacpp`/`vllm`, not `ollama`/`anthropic`/`gemini`.

`omop-llm` depends on `oa-configurator` for config resolution. Two entry points: 
1. `build_model_backend(provider, model, ...)` takes plain keyword arguments directly, and 
2. `build_model_backend_from_resolved(resolved)` takes an `oa_configurator.ResolvedModel` (from `Resolver(stack).resolve_model(name)`) and does the field mapping for you. 

`omop-llm` has no `PackageConfigBase` subclass of its own: it has no inherent specific model it needs. Each real consumer declares its own plain string field (e.g. `embedding_model: str = "embed-default"`) naming a `[models.*]` entry, and resolves it itself.

## What it deliberately does not do

- Install, launch, or manage any inference server (`ollama`, `llama-server`, `vllm`); that is Docker Compose / TRE deployment's job.
- Guess capabilities from a model name or provider string; see the capability model above.
- Reimplement `instructor`'s validate-and-retry loop, or any-llm's own per-provider wire translation; both are used directly, not duplicated.
- Own configuration parsing, TOML tables, or secrets; that is `oa-configurator`'s job.

## Documentation overview

- [Installation](usage/installation.md)
- [Asymmetric Embeddings](usage/asymmetric-embeddings.md)
- [Providers](providers.md)
- [API Reference](reference.md)
