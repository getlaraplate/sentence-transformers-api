# sentence-api

Small HTTP service (Flask, served by gunicorn) that serves embeddings and
cross-encoder reranking through [sentence-transformers](https://www.sbert.net/).
Used by the Laraplate stack: the client sends the active profile's
`service_model` in every request, so the default model is only used when a
request does not specify `model`. A request may only name an allowed model
(see [Security](#security)).

## Production readiness

**Status: experimental.** Fine for development, tests and evaluation on a
trusted network. Not ready for production yet: the gaps below are open, and the
status changes to `production` only when all of them are closed. `GET /health`
reports the same status in `readiness`.

**Tested on:** unit tests with fake models (no model download), run by hand.
Earlier releases ran on the Laraplate test container (Debian LXC, 4 GB RAM,
2 vCPU, CPU only); the hardened release (gunicorn, unprivileged user, keys,
allow-lists, limits) has not been verified there yet. **GPU:** not tested.

**Known gaps:**

- The documented install shares one virtualenv (`/opt/ai-env`) with
  `whisper-api`: upgrading one service can break the other.
- No container image, no tagged release, no changelog.
- Models load at startup with no separate readiness check: a process still
  loading looks alive to a load balancer or a healthcheck.
- No backpressure: when every thread is busy, requests wait without limit.
- `500` answers include the text of the exception.
- No metrics and no request id in the logs.
- No continuous integration: the tests run only by hand.
- No machine-readable API description and no service version in `/health`.
- Plain HTTP: TLS needs your own reverse proxy (see [Security](#security)).

**Default models:**

| Model | Use | Licence (model card) |
| ----- | --- | -------------------- |
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | `/embed` | MIT |
| [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | `/score` | Apache-2.0; trained on [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) (Apache-2.0), a translation of [MS MARCO](https://microsoft.github.io/msmarco/) |

The licence of a model is the one on its model card; a model you add to an
allow-list brings its own. The MS MARCO site states that its datasets "are
intended for non-commercial research purposes only": check whether that matters
for your use of the default reranker, or set another one with
`CROSS_ENCODER_MODEL`.

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
| `EMBEDDING_ALLOWED_MODELS` | *(empty)*                          | Comma-separated models a request may name, in addition to `EMBEDDING_MODEL`. `*` allows any model (development only) |
| `EMBEDDING_HOST`         | `0.0.0.0`                            | Listening address                             |
| `EMBEDDING_PORT`         | `8000`                               | Listening port                                |
| `EMBEDDING_MAX_REQUEST_MB` | `16`                               | Maximum request body, in MB, on every endpoint (`413` beyond) |
| `EMBEDDING_MAX_TEXTS`    | `256`                                | Maximum texts per `/embed` request (the Laraplate client sends `SENTENCE_TRANSFORMERS_BATCH_SIZE`, default 32) |
| `EMBEDDING_MAX_TEXT_CHARS` | `100000`                           | Maximum characters per `/embed` text and per `/score` query or text |
| `SENTENCE_TRANSFORMERS_API_KEY` | *(empty)*                     | Bearer key required on `/embed`, `/score` and `/models`. Several keys may be given, comma-separated. Empty: the service is open (trusted network or development only). Same name as on the Laraplate side |
| `CROSS_ENCODER_MODEL`    | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Default rerank model for `/score`, loaded at startup. Empty string: reranking off, `/score` answers 503 unless the request names a `model` |
| `CROSS_ENCODER_ALLOWED_MODELS` | *(empty)*                      | Comma-separated rerank models a request may name, in addition to `CROSS_ENCODER_MODEL`. `*` allows any model (development only) |
| `CROSS_ENCODER_MODEL_CACHE` | `1`                               | How many rerank models to keep resident (LRU) |
| `CROSS_ENCODER_MAX_LENGTH`  | `512`                             | Maximum tokens per pair (the rest is truncated) |
| `CROSS_ENCODER_MAX_PAIRS`   | `64`                              | Maximum pairs per request (the Laraplate client sends at most 64) |

Models are downloaded automatically from Hugging Face on first use and are not
versioned in the repository. Only the default and the allowed models can be
downloaded (see [Security](#security)).

## Running

For local development, Flask's built-in server:

```bash
python sentence-api.py
```

In production, gunicorn with the settings in [`gunicorn.conf.py`](gunicorn.conf.py):

```bash
gunicorn -c gunicorn.conf.py sentence-api:app
```

Both listen on `EMBEDDING_HOST` (default `0.0.0.0`), on port `EMBEDDING_PORT`
(default `8000`).

gunicorn runs **one worker process with 4 threads**: each worker process would
load its own copy of every model, so concurrency comes from threads instead
(torch releases the GIL while it computes; the model caches are guarded by a
lock). Override any setting with `GUNICORN_CMD_ARGS`, e.g.
`GUNICORN_CMD_ARGS="--threads 8"`; more workers multiply the memory used by the
models.

The default model is loaded at startup: if it cannot be downloaded or used, the
process exits immediately instead of starting and answering 500.

## Security

- **Authentication.** When `SENTENCE_TRANSFORMERS_API_KEY` is set, `/embed`,
  `/score` and `/models` require `Authorization: Bearer <key>`, the header the
  Laraplate client sends with `SENTENCE_TRANSFORMERS_API_KEY` (or
  `CROSS_ENCODER_API_KEY` for `/score`). A missing or wrong key answers `401`
  `{"error": "unauthorized"}`. To accept a distinct `CROSS_ENCODER_API_KEY`, list
  both keys, comma-separated. `/health` stays open for monitoring. Without a key
  the service is open: run it like that only on a trusted network.
- **Model allow-list.** Loading a model downloads it from Hugging Face, so a
  request may only name the default model or one listed in
  `EMBEDDING_ALLOWED_MODELS` / `CROSS_ENCODER_ALLOWED_MODELS`; any other answers
  `400` without loading anything. When a Laraplate profile uses another
  `service_model` than `EMBEDDING_MODEL`, add it to `EMBEDDING_ALLOWED_MODELS`.
- **Request limits.** Bodies larger than `EMBEDDING_MAX_REQUEST_MB` answer `413`;
  more than `EMBEDDING_MAX_TEXTS` texts, or a text longer than
  `EMBEDDING_MAX_TEXT_CHARS`, answer `400`.
- **Process.** The systemd unit runs gunicorn as the unprivileged
  `sentence-api` user (see [Deploy](#deploy-systemd)).
- **Transport.** The service speaks plain HTTP. Outside a private network, put
  it behind a TLS-terminating reverse proxy, or bind it to `127.0.0.1` with
  `EMBEDDING_HOST`.

## API

### `GET /health`

Service status and loaded models. Always open, even when a key is set.

```json
{
  "status": "healthy",
  "readiness": "experimental",
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

Discovery endpoint: the default, the models a request may name
(`allowed_models`, `null` when any is allowed), the models currently resident,
the text limits and the families that support `input_type`, so the client knows
when `input_type` matters.

```json
{
  "default_model": "intfloat/multilingual-e5-small",
  "allowed_models": ["intfloat/multilingual-e5-small"],
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "max_texts": 256,
  "max_text_chars": 100000,
  "input_types": ["query", "passage"],
  "prefix_families": {
    "e5": { "query": "query: ", "passage": "passage: " },
    "nomic": { "query": "search_query: ", "passage": "search_document: " }
  },
  "reranker": {
    "default_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "allowed_models": ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"],
    "loaded_models": ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"],
    "model_cache": 1,
    "max_length": 512,
    "max_pairs": 64
  }
}
```

### `POST /embed`

Generates embeddings. JSON body:

- `text` (string) **or** `texts` (array of strings), required, at most
  `EMBEDDING_MAX_TEXTS` texts of at most `EMBEDDING_MAX_TEXT_CHARS` characters
- `model` (string, optional), overrides the default model; must be allowed
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
  -H "Authorization: Bearer $KEY" \
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

Errors: `400` for a malformed body, too many or too long texts, or a model that
is not allowed, `401` without a valid key, `413` for a body over the limit, `500`
when the model fails.

### `POST /score`

Reranking with a cross-encoder: scores the relevance of each query-document
pair. JSON body:

- `pairs` (array of `{"query": string, "text": string}`), required, at most
  `CROSS_ENCODER_MAX_PAIRS`, each query and text at most
  `EMBEDDING_MAX_TEXT_CHARS` characters
- `model` (string, optional), overrides the default model; must be allowed

The model must be a **cross-encoder** (a model trained for reranking, not an
embedding model) and, for non-English content, multilingual. The server passes
the sigmoid explicitly, so scores are always in `[0, 1]`, even for checkpoints
that return logits by default.

```bash
curl -s http://localhost:8000/score \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"pairs": [{"query": "orari di apertura", "text": "Il museo è aperto dalle 9 alle 18"}]}'
```

Response:

```json
{ "model": "<model>", "scores": [0.93] }
```

Errors: `400` for a malformed body, too many or too long pairs, or a model that
is not allowed, `401` without a valid key, `413` for a body over the limit, `503`
when there is no model (`CROSS_ENCODER_MODEL` empty and no `model` in the
request), `500` when the model does not return one score per pair. The Laraplate client treats any error
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
in `/opt/sentence-api`, and runs gunicorn as the dedicated unprivileged user
`sentence-api`. Models are cached in `/var/cache/sentence-api` (`HF_HOME`), which
systemd creates and hands to that user.

Create the user once (no home, no login shell), and keep the code readable but
not writable by it:

```bash
sudo useradd --system --user-group --no-create-home --shell /usr/sbin/nologin sentence-api
sudo chown -R root:root /opt/sentence-api
```

Put the key, and any other local setting, in `/opt/sentence-api/.env`. systemd
reads the file as root before dropping privileges, so it can stay root-only:

```bash
printf 'SENTENCE_TRANSFORMERS_API_KEY=%s\n' "$(openssl rand -hex 24)" | sudo tee /opt/sentence-api/.env > /dev/null
sudo chmod 600 /opt/sentence-api/.env
```

Then install the unit:

```bash
sudo /opt/ai-env/bin/pip install -r /opt/sentence-api/requirements.txt
sudo cp /opt/sentence-api/sentence-transformers.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentence-transformers.service
```

Set the same key as `SENTENCE_TRANSFORMERS_API_KEY` in the Laraplate `.env`.

Reranking is on with the default model. To configure it, set
`CROSS_ENCODER_MODEL=<model>` in `.env` to change the rerank model, or
`CROSS_ENCODER_MODEL=` to turn it off, then restart.

Upgrading from a unit that ran as `root`: models cached in
`/root/.cache/huggingface` are not visible to the new user and are downloaded
again into `/var/cache/sentence-api` on the first start. To skip the download,
copy them before starting:

```bash
sudo mkdir -p /var/cache/sentence-api
sudo cp -a /root/.cache/huggingface/. /var/cache/sentence-api/
sudo chown -R sentence-api:sentence-api /var/cache/sentence-api
```

The commented `PrivateTmp`, `ProtectSystem` and `ProtectHome` lines in the unit
add sandboxing where mount namespaces are available (not in every unprivileged
LXC container).

## License

Licensed under the [Apache License 2.0](LICENSE).
