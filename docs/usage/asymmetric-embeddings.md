# Asymmetric Embeddings { data-toc-label="Asymmetric Embeddings" }

## What are asymmetric embedding models?

Most general-purpose embedding models (e.g. `text-embedding-3-small`) produce vectors in a symmetric space: the same transformation is applied whether you are indexing a document or submitting a search query.

**Asymmetric models**, such as [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [E5](https://huggingface.co/intfloat/e5-large-v2), and [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5), are trained with *task-specific prefixes* prepended to the input. The model's training objective explicitly separates the representation space for documents being indexed from the space for queries being searched. Sending text without the correct prefix does not raise an error, but similarity scores degrade substantially and silently — this is a correctness footgun, not a performance one.

!!! tip "Prefix examples by model family"
    | Model | Document prefix | Query prefix |
    |---|---|---|
    | `nomic-embed-text:v1.5` | `search_document: ` | `search_query: ` |
    | `e5-large-v2` | `passage: ` | `query: ` |
    | `bge-large-en-v1.5` | *(none)* | `Represent this sentence for searching relevant passages: ` |

    Always check the model card: task prefixes are model-specific and can change between versions.

## `EmbeddingRole`: the two roles a text can play

`omop_llm.EmbeddingRole` is a `StrEnum` with two members: `DOCUMENT` (text being indexed) and `QUERY` (text used to search). It exists because the prefix convention is a property of the *model*, not of whatever domain is calling it — every consumer of a given asymmetric model shares the same two roles and the same prefix strings, so the concept lives here rather than being reinvented per consumer package.

```python
from omop_llm import EmbeddingRole, ModelBackend

# Indexing
doc_vectors = model_backend.embed_texts(["Hypertension", "Diabetes"], role=EmbeddingRole.DOCUMENT)

# Searching
query_vectors = model_backend.embed_texts(["high blood pressure"], role=EmbeddingRole.QUERY)
```

Passing `role=` is optional. Omit it (or leave it `None`) for a symmetric model, or when you don't want prefix application for some other reason — `embed_texts`/`async_embed_texts` pass the text through unchanged when `role` is `None`.

## Where the prefix values come from

Prefix strings are configured once per model, not per caller. Two ways to supply them:

1. **Direct `configuration` dict**, when calling `build_model_backend` with plain keyword arguments:

    ```python
    from omop_llm import build_model_backend

    backend = build_model_backend(
        "ollama",
        "nomic-embed-text:v1.5",
        base_url="http://localhost:11434",
        configuration={
            "document_prefix": "search_document: ",
            "query_prefix": "search_query: ",
        },
    )
    ```

2. **`oa-configurator`'s `[models.*]` entry**, when calling `build_model_backend_from_resolved(resolved)`. `ModelConfig` has typed `document_prefix`/`query_prefix` fields (alongside `embedding_dim`) specifically for this — see `oa-configurator`'s [config reference](https://AustralianCancerDataNetwork.github.io/OA_Configurator/config-reference/#modelsname) for the TOML shape and `omop-config models add`/`list` for managing it via CLI. `build_model_backend_from_resolved` folds these typed fields into the `configuration` dict before constructing the backend, so the mechanism is identical either way — only the source of the values differs.

Either way, the resolved `configuration` dict is what `apply_embedding_prefix`/`embed_texts(role=...)` actually reads at call time; `ModelBackend` itself doesn't care whether the values came from a literal dict or a resolved config entry.

## Construction-time sanity check

`build_model_backend`/`build_model_backend_from_resolved` call `warn_if_prefixes_look_wrong` once, at construction, whenever the resolved backend is embedding-capable. It logs (never raises) two independent things:

- **Missing**: no `document_prefix`/`query_prefix` configured at all. Fine for a symmetric model; worth double-checking for an asymmetric one.
- **Unrecognized**: a prefix is configured, but doesn't match anything in `omop_llm.embeddings.KNOWN_EMBEDDING_PREFIXES` — a small `frozenset` of common conventions (`"search_document: "`, `"passage: "`, etc.), deliberately *not* an exhaustive per-model lookup table (the space of models is unbounded and grows constantly). An unrecognized prefix isn't necessarily wrong — it's a nudge to double-check against the model card, not a rejection.

Neither warning blocks construction. There is no validation that a prefix is *correct* for a given model — that isn't knowable from the string alone.
