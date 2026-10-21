# sentence-transformers-api: production readiness

**Status:** Draft, 2026-10-20. Phase 0 delivered on 2026-10-20 (6fdac84); Phase 1 delivered on 2026-10-21 by `../plans/2026-10-20-phase-1-runtime-hardening.md`.
**Builds on:** the hardening of 2026-10-20 (18678e7).
**Siblings:** the same work for [`whisper-api`](https://github.com/getlaraplate/whisper-api/blob/main/docs/superpowers/specs/2026-10-20-production-readiness-design.md), and the Laraplate client side in [`laraplate`](https://github.com/getlaraplate/laraplate/blob/master/docs/superpowers/specs/2026-10-20-self-hosted-ai-services-client-design.md). The three specs share the decision and requirement numbers (D, R, E), so a number means the same thing in each.
**History:** split on 2026-10-20 from one spec at the root of the Laraplate workspace, so that each repository owns the design of its own work.

## 1. Goal

This service is part of the Laraplate offer as **free, self-hosted
infrastructure that the customer installs**. Either it is fit for production on
the customer's servers, or it says plainly that it is not. Nobody should run it
in production believing it is ready when it is not.

## 2. Decisions

| # | Decision | Source |
|---|---|---|
| D1 | The customer installs and runs the service. Laraplate does not host it for customers. Multi-tenancy, quotas and billing are out of scope. | Founder, 2026-10-20 |
| D2 | The service stays free (Apache-2.0) and is not sold. Whether the repository stays public or becomes private is decided separately and does not change this spec. | Founder, 2026-10-20 |
| D3 | CPU is the baseline. GPU is supported where it can be built and tested; where it cannot, the docs say so. | Founder, 2026-10-20 |
| D4 | Models follow the latest release available on Hugging Face; no revision is pinned by default. A model whose new release breaks compatibility gets pinned, together with provisioning the weights ahead of time and running offline (R10). Python dependencies follow the same rule: the newest release that passes the tests. | Founder, 2026-10-20 |
| D5 | The honesty requirement comes first: the readiness label (section 4) ships before any other work in this spec. | This spec |

## 3. State on 2026-10-20

Done by the hardening: Bearer-key auth (open when no key is set, `/health`
always open, several keys accepted), model allow-lists
(`EMBEDDING_ALLOWED_MODELS`, `CROSS_ENCODER_ALLOWED_MODELS`), limits on body
size, number of texts and text length, gunicorn (one process, `gthread`) instead
of Flask's development server, a systemd unit running as the unprivileged
`sentence-api` user, unit tests with fake models.

Still missing for production:

1. **Shared environment.** The documented install shares one virtualenv (`/opt/ai-env`) with `whisper-api`: torch on one side, CTranslate2 on the other. Upgrading one can break the other.
2. **No packaging.** No container image, no tagged release, no changelog.
3. **No readiness.** Models load when the module is imported, so a process that is still loading looks alive to a load balancer or a container healthcheck.
4. **No backpressure.** When every thread is busy, requests wait without limit.
5. **Errors leak.** `500` answers return `str(exc)`.
6. **No observability.** No metrics, no request id, logs are gunicorn's access log only.
7. **No CI.** The tests exist; nothing runs them on push.
8. **No contract.** No machine-readable API description, no version in `/health`.
9. **Default model licences** were not stated (done in Phase 0, see R9).

## 4. Readiness label

The README starts with a **Production readiness** section that states:

- **Status:** `experimental` or `production`.
- What is tested and on which hardware (CPU model and RAM, GPU model and CUDA version when supported).
- What is known not to work or not to be ready, listed in the README itself, and a link to this spec.
- The licences of the default models (R9).

Rules:

- Status is `experimental` until every requirement of section 5 is done.
- It becomes `production` in the same change that closes the last requirement of section 5, and only then.
- GPU support has its own line. It reads `supported` only for a configuration that has been run with a real model, on real hardware, for a given release (R1).
- The Laraplate install guide (`laraplate/Modules/AI/docs/SENTENCE_TRANSFORMERS_INSTALLATION.md`) repeats the status in its first lines and links the README section.
- `/health` reports the status as `"readiness": "experimental" | "production"`.

## 5. Requirements for `production`

### R1. Packaging and releases

- A virtualenv or an image of its own, never shared with another service. The systemd install moves from `/opt/ai-env` to `/opt/sentence-api/.venv`.
- Container images published to GitHub Container Registry for each tag: a CPU image, and a CUDA image where D3 allows. A `compose.yaml` of reference.
- The systemd install stays supported for bare metal and LXC.
- Semantic version tags, a `CHANGELOG.md`, a GitHub release per tag.
- The CUDA image records the CUDA and cuDNN versions it was built against and the torch build it uses.

### R2. Startup, liveness and readiness

- Models load in a background thread after the server starts, not at import.
- `GET /health` answers as soon as the process runs (liveness).
- `GET /ready` answers `503` until the default embedding model and the default reranker are loaded, then `200`. The container healthcheck and the docs use `/ready`.
- `/embed` and `/score` answer `503` with `Retry-After` while the service is not ready.
- A default model that fails to load makes the process exit with a non-zero code, so systemd or the container runtime restarts it. *Amended 2026-10-21:* `/ready` does not report the error, since a process that is exiting cannot answer; the reason is in the log (exit code 3, gunicorn's boot error).

### R3. Backpressure and resource limits

- A bounded number of requests in flight (configurable). Beyond it: `503` with `Retry-After`, not an unbounded wait.
  *Delivered 2026-10-21:* `EMBEDDING_MAX_IN_FLIGHT`, default 2 after measuring the memory (peak 1.9 GB with two requests at a time, 3.1 GB with six).
- The systemd unit sets `MemoryMax`; the compose file sets a memory limit. The docs give the figures for the default models and for `EMBEDDING_MODEL_CACHE` above 1.

### R4. Errors

- `5xx` answers carry a generic message and the request id, never the exception text. The exception goes to the log.
- Every answer carries `X-Request-Id`: the one the client sent, or a new one. The Laraplate client side is in the sibling spec.

### R5. Observability

- `GET /metrics` in Prometheus format: requests and latency per endpoint and status, requests in flight and rejected, model load time, models resident. Behind the API key when one is set, or on a separate port.
- Optional JSON logs.
- Logs never contain texts or queries: they are customer data.
- New dependency to approve: `prometheus-client` (Q2).

### R6. Continuous integration

- GitHub Actions on every push: unit tests, `ruff`, `pip-audit`.
- Dependabot for pip, Docker and Actions. Updates merge when the tests pass (D4).
- A nightly job with small real models (an embedding model and a cross-encoder) that exercises the HTTP contract end to end.
- A release workflow that builds and pushes the images on a tag.

### R7. Versions and contract

- `/health` adds `version` (the release) and `api_version` (the HTTP contract, starting at `1`) next to `readiness`.
- An OpenAPI document of the current endpoints, kept in the repository. The Laraplate client tests against it (sibling spec).

### R8. Transport

- The README gives a reverse proxy configuration of reference for TLS (Caddy and nginx). The compose file includes an optional Caddy service.
- The default listening address stays `0.0.0.0` for container use; the bare-metal docs recommend `127.0.0.1` behind the proxy.

### R9. Model licences

Delivered in Phase 0: the README lists the default models with their licence and links each model card.

| Default | Model card licence | Note |
|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Apache-2.0 | Trained on mMARCO (Apache-2.0), a translation of MS MARCO. The MS MARCO site states that its datasets "are intended for non-commercial research purposes only". |

The service is free, but customers run Laraplate for their business. The
reranker default needs a decision (Q1): keep it with the MS MARCO terms stated,
as the README does now, or move the default to a reranker with no such upstream
terms. `BAAI/bge-reranker-v2-m3` (Apache-2.0, multilingual, about 568M
parameters, so slower on CPU than the current one) is a candidate whose training
data must be checked first.

### R10. Model updates (D4)

- No revision pinned by default: a model loads at its latest release.
- At load, the service logs the resolved revision (the commit of the Hugging Face cache) and `/models` reports it per resident model, so a regression can be traced to a model release.
- Laraplate already refuses vectors of the wrong length and a model other than the one requested.
- When a model release breaks compatibility: an env var pins the revision (`SentenceTransformer` and `CrossEncoder` accept a `revision` argument), and the docs describe provisioning the weights ahead of time and running with `HF_HUB_OFFLINE=1`. This is the fallback in D4, not the default.

## 6. Evolutions

They do not gate `production`.

### E1. Standard protocols (highest leverage)

- Next to the current endpoints, which stay: `POST /v1/embeddings` in the OpenAI shape and `POST /rerank` in the Text Embeddings Inference shape.
- With the Laraplate client speaking the same protocols (sibling spec), a customer can run this service, Text Embeddings Inference (Apache-2.0) or a cloud API with only `.env` changes. Whether to keep, open or replace the service becomes a reversible choice.

### E3. CPU performance

- ONNX or OpenVINO backend and quantization through `sentence-transformers` (env `EMBEDDING_BACKEND`), after a benchmark on the reference container (4 GB, 2 vCPU).
- Micro-batching of concurrent `/embed` and `/score` requests into one model call.
- A benchmark table in the README: hardware, model, throughput, latency.

### E4. GPU

- The CUDA image (R1) with defaults that use the GPU when present.
- Docs for the NVIDIA container toolkit and for GPU passthrough to an LXC container.
- D3 applies: a GPU configuration that has not been run reads `not tested` in the readiness section.

### E5. Keys with scopes

- A keys file with one entry per key: scopes (`embed`, `score`, `models`, `metrics`) and a rate limit per key. Keys stored as hashes.
- `SENTENCE_TRANSFORMERS_API_KEY` keeps working as keys with every scope.

## 7. Out of scope

- Hosting the service for customers, multi-tenancy, quotas, billing (D1).
- Selling the service or its support (D2).
- Pinning model revisions by default (D4).

## 8. Order of work

| Phase | Content |
|---|---|
| 0 | Readiness section `experimental`, model licences, `readiness` in `/health`. **Delivered 2026-10-20** (6fdac84) |
| 1 | R2, R3, R4, R7, R10. **Delivered 2026-10-21** (`../plans/2026-10-20-phase-1-runtime-hardening.md`) |
| 2 | R6, R1, R8, R5 |
| 3 | Status `production` when section 5 is complete |
| 4 | E1 |
| 5 | E3, E4 |
| 6 | E5 |

**Phase 0 delivery (2026-10-20):** the README opens with a Production readiness
section (status `experimental`, what was tested, known gaps, default model
licences) and `/health` reports `readiness`. The Laraplate guide opens with the
status and links that section (`laraplate-ai` c2530f0).

## 9. Open questions

- **Q1. Reranker default.** Keep `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` with the MS MARCO terms stated, or move to a reranker without non-commercial upstream terms (R9)?
- **Q2. New dependencies.** `prometheus-client` (R5); GitHub Container Registry for the images (R1); `ruff` and `pip-audit` in CI only (R6).
- **Q3. GPU hardware.** Which GPU, if any, is available to run the CUDA configuration before a release (D3, E4)?
