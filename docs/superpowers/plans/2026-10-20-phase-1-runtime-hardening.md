---
status: closed
created_on: 2026-10-20
---
# Phase 1 runtime hardening. Implementation Plan

> **For agentic workers:** execute task by task. Steps use checkbox (`- [ ]`) syntax; tick each step when it is done, in the same work block. A step deliberately not done is `- [-]` with its reason on the same line.

**Status:** Closed (2026-10-21): delivered on `main`, see "Delivery status" at the end.

**Goal:** the service starts answering at once and says when it is ready, refuses work it cannot take instead of queueing it without limit, never leaks an exception, carries a request id, declares its version and HTTP contract, and reports the revision of every model it loaded, with a way to pin one.

**Architecture:** all in `sentence-api.py` plus `gunicorn.conf.py`, `openapi.json` and `VERSION` at the repository root. A loader thread replaces the loading at import; a `before_request` chain runs auth, then the readiness gate, then the in-flight limit; an `after_request` hook sets `X-Request-Id`; one error handler turns any unexpected exception into a generic `500`.

**Tech Stack:** Python 3.11+, Flask 3.1, gunicorn 26.2 (`gthread`), sentence-transformers, `huggingface_hub` (already installed by sentence-transformers). No new dependency.

**Spec:** `../specs/2026-10-20-production-readiness-design.md`, section 5: R2, R3, R4, R7, R10 (Phase 1 of section 8).

## Global Constraints

- Apache-2.0 repository. No new dependency in `requirements.txt`.
- The HTTP contract the Laraplate client relies on stays: same endpoints, request and response shapes, status codes, plus the new `503` answers and the new fields.
- Tests are `unittest` with the fakes of `tests/service_case.py`; no model download in the unit suite. `python -m unittest discover -s tests` passes before every commit, with `-W error::ResourceWarning`.
- No text, query or model input in any log line or error answer.
- Conventional commits, trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, identity `swolley <myentity@inwind.it>`. Never push without the user's OK; never touch the stack's submodule pointers.

## Decisions taken in this plan

- **P1. A default model that fails to load ends the process** with exit code 3, gunicorn's worker boot error, so gunicorn stops and systemd (or the container runtime) restarts it after `RestartSec`. The spec's "`/ready` reports the error" is dropped: a process that is exiting cannot answer. The error goes to the log at ERROR.
- **P2. `/ready` is open like `/health`**, so healthchecks need no key. It answers `{"status": "loading"}` with `503`, or `{"status": "ready"}` with `200`.
- **P3. Readiness gate.** While loading, `/embed` and `/score` answer `503` `{"error": "loading models"}` with `Retry-After: 10`. `/models` answers as usual. The gate runs after the auth check, so an unauthorized request still gets `401`.
- **P4. In-flight limit.** `EMBEDDING_MAX_IN_FLIGHT` (default 6) bounds the `/embed` and `/score` requests being served at once; one more answers `503` `{"error": "busy"}` with `Retry-After: 2`, at once, without waiting. gunicorn moves to 8 threads, so `/health`, `/ready` and the refusals always have a thread. gunicorn's `worker_connections` drops to 32 to bound the connections it queues.
- **P5. Request id.** An incoming `X-Request-Id` matching `^[A-Za-z0-9._-]{1,128}$` is kept; otherwise a new one is generated (`uuid4().hex`). Every answer carries it. gunicorn's access log prints it.
- **P6. `500` answers** are `{"error": "internal error", "request_id": "<id>"}`. The exception goes to the log with the request id and the endpoint. This includes the "not one score per pair" case of `/score`, which loses its specific message in the answer and keeps it in the log.
- **P7. Version.** A `VERSION` file at the root holds the release (`0.1.0` in this plan; tags come with R1 in Phase 2). `/health` adds `version` and `api_version: 1`.
- **P8. OpenAPI.** `openapi.json` (OpenAPI 3.1), hand-written, every endpoint with request and response schemas and one example each. A test checks that the routes of the app and the paths of the document match, and that the answers of the test client carry the required fields of their schema. No validator library: a small recursive required-and-type check in the tests.
- **P9. Revision.** After loading a model, the service reads the revision from the Hugging Face cache (`huggingface_hub.try_to_load_from_cache(name, "config.json")`, whose path holds `snapshots/<commit>/`), logs it, and `/models` reports it per resident model as `revisions: {name: commit | null}`. `null` for a local path or an unknown cache layout.
- **P10. Pinning.** A configured model name may carry `@<revision>`: `EMBEDDING_MODEL`, `CROSS_ENCODER_MODEL` and the allow-list entries. The request names the model without the suffix; the answer's `model` is also without it, as the Laraplate client compares it with what it asked for. The revision is passed to `SentenceTransformer` and `CrossEncoder` as `revision=`.

## Review Focus

- A request that arrives during loading with a valid key gets `503`, with a wrong key `401` (P3).
- The in-flight counter is released on every path: success, `400`, `500`, client disconnect (P4). Test the release after a `500`.
- No exception text in any answer, including `/score`'s model-shape errors (P6).
- `model@revision` never leaks into the answer's `model` (P10), otherwise the Laraplate client refuses the vectors.
- The loader thread and the lazy loads of allowed models share the cache lock without deadlock.

---

### Task 1: Background loading and `/ready` (R2)

**Files:** modify `sentence-api.py`, `tests/service_case.py`; create `tests/test_readiness.py`.

- [x] **Failing tests.** `/health` answers `200` while the default models are still loading (a fake model whose constructor blocks on an event); `/ready` answers `503` `{"status": "loading"}` then `200` `{"status": "ready"}`; `/embed` and `/score` answer `503` with `Retry-After` while loading, `401` without a key when one is set; a default model whose constructor raises makes the loader call the exit function with code 3 (inject the exit function so the test does not end the process).
- [x] **Implement.** Move the warm-up from import time into a loader thread started at import; a readiness flag guarded by the cache lock; `/ready`; the gate in `before_request` after auth. `service_case.py` gains a helper that waits for the loader so the existing tests keep their ready service.
- [x] **Run** the suite; commit `feat: load models in the background and add /ready`.

### Task 2: In-flight limit (R3)

**Files:** modify `sentence-api.py`, `gunicorn.conf.py`; create `tests/test_backpressure.py`.

- [x] **Failing tests.** With `EMBEDDING_MAX_IN_FLIGHT=1` and a fake encode blocked on an event, a second `/embed` from another thread answers `503` `{"error": "busy"}` with `Retry-After`; after the first returns, a new request passes. The slot is released after a `500` (a fake encode that raises). `/health` and `/ready` are not counted.
- [x] **Implement** P4: a bounded semaphore acquired without blocking in `before_request` for `/embed` and `/score`, released in `teardown_request`. `gunicorn.conf.py`: `threads = 8`, `worker_connections = 32`.
- [x] **Measure memory.** In a scratch virtualenv with the pinned requirements, load the default embedding model and reranker (downloads them) and read the resident memory after one `/embed` and one `/score`; repeat with `EMBEDDING_MODEL_CACHE=2` and a second allowed model. Write the figures in the README and set `MemoryMax` in `sentence-transformers.service` at about 1.5 times the default figure, with a comment on how to raise it.
- [x] **Run** the suite; commit `feat: refuse work beyond the in-flight limit`.

> **Finding (2026-10-21):** peak 1.6 GB with one request at a time, 1.9 GB with two, 3.1 GB with six. The default of `EMBEDDING_MAX_IN_FLIGHT` drops from 6 (P4) to 2, and `MemoryMax` is 3G. Figures in the README, section Memory.

### Task 3: Request id and generic errors (R4)

**Files:** modify `sentence-api.py`, `gunicorn.conf.py`; create `tests/test_errors.py`.

- [x] **Failing tests.** A valid incoming `X-Request-Id` is echoed; an invalid one (too long, a space, a newline) is replaced by a 32-hex id; every endpoint, including `401`, `413` and `503`, carries the header. A fake encode raising `RuntimeError("secret path /x")` gives `500` `{"error": "internal error", "request_id": ...}` and the answer does not contain `secret`; the log record (captured with `assertLogs`) holds the request id and the exception. `/score` with a model returning two scores per pair: `500`, generic message.
- [x] **Implement** P5 and P6: request id in `before_request` (first in the chain), header in `after_request`, a `@app.errorhandler(Exception)` that re-raises `HTTPException` and logs the rest; drop the `try`/`except Exception` blocks of the views. gunicorn `access_log_format` with `%({x-request-id}o)s`.
- [x] **Run** the suite; commit `feat: add request ids and stop leaking exceptions`.

### Task 4: Version and OpenAPI document (R7)

**Files:** create `VERSION`, `openapi.json`, `tests/test_contract.py`; modify `sentence-api.py`.

- [x] **Failing tests.** `/health` carries `version` equal to the `VERSION` file and `api_version` `1`; the set of (path, method) of `app.url_map` without `static` equals the one of `openapi.json`; for each endpoint, the answers of the test client in the success case and in each documented error case carry the required fields, with the documented JSON types, of their schema.
- [x] **Implement** P7 and P8. The document describes `/health`, `/ready`, `/models`, `/embed`, `/score`, the Bearer scheme, the error answers (`400`, `401`, `413`, `500`, `503`) and the `X-Request-Id` header.
- [x] **Run** the suite; commit `feat: declare the version and the HTTP contract`.

### Task 5: Model revisions and pinning (R10)

**Files:** modify `sentence-api.py`, `tests/service_case.py`; create `tests/test_revisions.py`.

- [x] **Verify** that `huggingface_hub.try_to_load_from_cache` in the version installed by the pinned sentence-transformers returns a path under `snapshots/<commit>/`, and that `SentenceTransformer` and `CrossEncoder` accept `revision=`. If either differs, record the finding in this plan and adapt P9 or P10.
- [x] **Failing tests.** The fakes record the `revision` they are given. `EMBEDDING_MODEL=org/m@abc` loads `org/m` with `revision="abc"`; a request naming `org/m` is served by it and the answer's `model` is `org/m`; a request naming `org/m@abc` is refused (`400`, not allowed). `/models` reports `revisions` from a patched cache lookup, `null` when the lookup finds nothing. The same for `CROSS_ENCODER_MODEL` and the allow-lists.
- [x] **Implement** P9 and P10.
- [x] **Run** the suite; commit `feat: report model revisions and allow pinning one`.

> **Verified (2026-10-21):** huggingface_hub 1.33.0 returns `.../snapshots/<commit>/config.json`; sentence-transformers 5.6.0 `SentenceTransformer` and `CrossEncoder` accept `revision`. A real run with `EMBEDDING_MODEL=intfloat/multilingual-e5-small@614241f…` and `HF_HUB_OFFLINE=1` loads the pinned snapshot, answers `model` without the suffix and reports both revisions in `/models`.

### Task 6: Documentation and closing

**Files:** modify `README.md`, `../specs/2026-10-20-production-readiness-design.md`, this plan.

- [x] **Docs.** README: `/ready` and the healthcheck to use; `EMBEDDING_MAX_IN_FLIGHT`, the `503` answers and `Retry-After`; `X-Request-Id`; `version`, `api_version` and `openapi.json`; the `@revision` suffix, provisioning the weights ahead of time and `HF_HUB_OFFLINE=1`; the memory figures and `MemoryMax`. Update the Production readiness section: the gaps closed by this plan leave the list; the status stays `experimental` (Phase 2 is open).
- [x] **Spec.** Status line points to this plan and records Phase 1 delivered; amend R2 with P1.
- [x] **Close.** Delivery status with divergences, ticks, `status: closed`; commit `docs: close the phase 1 plan`.

## Delivery status (2026-10-21): shipped

All six tasks are built on `main`, commits `1f4f21f` to the closing commit. The service starts answering at
once and loads the default models in a background thread (`/ready`, `503` with `Retry-After` on `/embed` and
`/score` while loading, exit code 3 when a default model fails to load); refuses `/embed` and `/score` beyond
`EMBEDDING_MAX_IN_FLIGHT` with `503 busy`; carries `X-Request-Id` on every answer and in gunicorn's access log;
answers unexpected errors with a generic `500` and the request id; declares `version` (from `VERSION`,
`0.1.0`) and `api_version` (`1`) in `/health`, with `openapi.json` checked by the tests; and reports the
revision of every resident model in `/models`, with `name@revision` to pin one. 73 unit tests pass; a real run
under gunicorn with the default models confirmed readiness, the in-flight refusals, the request id in the
access log, the pinned revision offline and the memory figures.

Divergences from the plan text:

- **P4 default (Task 2).** `EMBEDDING_MAX_IN_FLIGHT` defaults to 2, not 6: measured peak 1.9 GB with two
  requests at a time and 3.1 GB with six, too much for the 4 GB reference container shared with
  `whisper-api`. `MemoryMax=3G` in the unit; figures in the README, section Memory.
- **P1 against the spec (Task 1).** As planned, `/ready` does not report a load error: the process exits.
  R2 of the spec is amended accordingly.
- **Request id test (Task 3).** The "invalid id" case uses a tab instead of a newline: the test client, like
  HTTP servers, refuses a newline in a header value. The pattern is checked with `fullmatch`, so a trailing
  newline could not pass anyway.
- **Logging.** The service logger gets a stream handler of its own (time, level, name), as gunicorn's error
  log does not carry application loggers by default.

**Documented in:** `README.md` (Production readiness, Configuration, Model updates, Running: Memory, Startup
and readiness; Security; API) and `openapi.json`.
