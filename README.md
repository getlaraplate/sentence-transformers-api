# sentence-api

Small HTTP service (Flask) that serves embeddings through
[sentence-transformers](https://www.sbert.net/). Used by the Laraplate stack:
the client sends the active profile's `service_model` in every request, so the
default model is only used when a request does not specify `model`.

## Requirements

- Python 3.11+ (`requirements.txt` pins `numpy==2.3.2`, which requires 3.11)
- The dependencies in [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (environment variables)

| Variable                 | Default                              | Description                                   |
| ------------------------ | ------------------------------------ | --------------------------------------------- |
| `EMBEDDING_MODEL`        | `intfloat/multilingual-e5-small`     | Model loaded at startup and used by default   |
| `EMBEDDING_MODEL_CACHE`  | `2`                                  | How many models to keep resident (LRU)        |
| `EMBEDDING_PORT`         | `8000`                               | Listening port                                |
| `CROSS_ENCODER_MODEL`    | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Default rerank model for `/score`, loaded at startup. Empty string: reranking off, `/score` answers 503 unless the request names a `model` |
| `CROSS_ENCODER_MODEL_CACHE` | `1`                               | How many rerank models to keep resident (LRU) |
| `CROSS_ENCODER_MAX_LENGTH`  | `512`                             | Maximum tokens per pair (the rest is truncated) |
| `CROSS_ENCODER_MAX_PAIRS`   | `64`                              | Maximum pairs per request (the Laraplate client sends at most 64) |

Models are downloaded automatically from Hugging Face on first use and are not
versioned in the repository.

## Running

```bash
python sentence-api.py
```

The service listens on `0.0.0.0`, on port `EMBEDDING_PORT` (default `8000`).

The default model is loaded at startup: if it cannot be downloaded or used, the
process exits immediately instead of starting and answering 500.

## API

### `GET /health`

Service status and loaded models.

```json
{
  "status": "healthy",
  "model": "intfloat/multilingual-e5-small",
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "reranker": {
    "default_model": null,
    "loaded_models": [],
    "model_cache": 1,
    "max_length": 512,
    "max_pairs": 64
  }
}
```

`reranker.default_model` set to `null` means the host does not rerank.

### `GET /models`

Discovery endpoint: the service loads any model on request, so there is no
fixed list of allowed models. It returns the default, the models currently
resident and the families that support `input_type`, so the client knows when
`input_type` matters.

```json
{
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "input_types": ["query", "passage"],
  "prefix_families": {
    "e5": { "query": "query: ", "passage": "passage: " },
    "nomic": { "query": "search_query: ", "passage": "search_document: " }
  }
}
```

### `POST /embed`

Generates embeddings. JSON body:

- `text` (string) **or** `texts` (array of strings), required
- `model` (string, optional), overrides the default model
- `input_type` (`"query"` | `"passage"`, optional), the **semantic** intent:
  `query` = searching, `passage` = indexing. The server guarantees the right
  prefix for the model (e.g. e5 → `query:` / `passage:`) **idempotently**: a text
  that already carries the prefix is kept, otherwise the prefix is added (no
  risk of `query: query: ...`). Ignored for models without a convention. When
  omitted, no prefix.
- `normalize_embeddings` (bool, optional, default `true`)

The client stays **model-agnostic**: it always sends the same shape and leaves
to the server the knowledge of how to talk to each model.

```bash
curl -s http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["ciao mondo", "hello world"], "input_type": "passage"}'
```

Response:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "input_type": "passage",
  "prefix_applied": true,
  "embeddings": [[...], [...]]
}
```

### `POST /score`

Reranking with a cross-encoder: scores the relevance of each query-document
pair. JSON body:

- `pairs` (array of `{"query": string, "text": string}`), required, at most
  `CROSS_ENCODER_MAX_PAIRS`
- `model` (string, optional), overrides the default model

The model must be a **cross-encoder** (a model trained for reranking, not an
embedding model) and, for non-English content, multilingual. The server passes
the sigmoid explicitly, so scores are always in `[0, 1]`, even for checkpoints
that return logits by default.

```bash
curl -s http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"pairs": [{"query": "orari di apertura", "text": "Il museo è aperto dalle 9 alle 18"}]}'
```

Response:

```json
{ "model": "<model>", "scores": [0.93] }
```

Errors: `400` for a malformed body or too many pairs, `503` when there is no
model (`CROSS_ENCODER_MODEL` empty and no `model` in the request), `500` when the
model does not return one score per pair. The Laraplate client treats any error
as "rerank not performed" and keeps the original order.

On the Laraplate side, the base URL is set with `CROSS_ENCODER_URL` (e.g.
`http://HOST:8000`, without `/score`, which the client appends). When it is not
set, `SENTENCE_TRANSFORMERS_URL` is used: reranking runs on the same service as
the embeddings. There is no default address. The key is `CROSS_ENCODER_API_KEY`,
falling back to `SENTENCE_TRANSFORMERS_API_KEY`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests replace `sentence-transformers` and `torch` with fakes: they download
no model and check what the service asks of the model and what it answers.

## Deploy (systemd)

The [`sentence-transformers.service`](sentence-transformers.service) file is the
unit used in production. It assumes the virtualenv in `/opt/ai-env` and the code
in `/opt/sentence-api`. Reranking is on with the default model. To configure it,
add `Environment=CROSS_ENCODER_MODEL=<model>` to the unit to change the rerank
model, or `Environment=CROSS_ENCODER_MODEL=` to turn it off, then restart.

```bash
sudo cp sentence-transformers.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentence-transformers.service
```

## License

Licensed under the [Apache License 2.0](LICENSE).
