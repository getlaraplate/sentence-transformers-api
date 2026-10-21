import hmac
import logging
import math
import os
import re
import threading
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request
from sentence_transformers import CrossEncoder, SentenceTransformer
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
logger = logging.getLogger("sentence_api")
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
    logger.setLevel(logging.INFO)

# An incoming X-Request-Id is kept when it looks like an id; otherwise one is made.
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}")


def env_list(name):
    """Comma-separated values of an environment variable, blanks dropped."""
    return [value.strip() for value in os.environ.get(name, "").split(",") if value.strip()]


def split_revision(configured):
    """Split a configured "name@revision" into (name, revision); revision is None when absent.

    Models load at their latest revision. A revision is pinned only when a model
    release breaks compatibility; requests and answers always use the bare name.
    """
    if not configured:
        return configured, None
    name, _, revision = configured.partition("@")
    return name, revision or None


# Bearer keys accepted on /embed, /score and /models. Several keys may be given,
# comma-separated, to rotate a key or to accept a distinct CROSS_ENCODER_API_KEY
# from the Laraplate client. No key means the service is open (trusted network or
# local development only). /health and /ready always stay open for monitoring.
API_KEYS = env_list("SENTENCE_TRANSFORMERS_API_KEY")
PUBLIC_PATHS = {"/health", "/ready"}
# Endpoints that need the default models: they answer 503 until those are loaded.
WORK_PATHS = {"/embed", "/score"}

# Production readiness of this release, reported by /health and stated in the
# README: "experimental" until every requirement for production is met.
READINESS = "experimental"
# The release, from the VERSION file next to this one, and the version of the HTTP
# contract described in openapi.json; a change that breaks a client raises it.
VERSION_FILE = Path(__file__).with_name("VERSION")
VERSION = VERSION_FILE.read_text().strip() if VERSION_FILE.is_file() else "unknown"
API_VERSION = 1

# Default model, and how many models to keep resident. The Laraplate client
# sends the active profile's service_model in each request, so this default is
# only used when a request omits "model".
DEFAULT_MODEL, _default_revision = split_revision(os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
MODEL_CACHE = max(1, int(os.environ.get("EMBEDDING_MODEL_CACHE", "2")))
HOST = os.environ.get("EMBEDDING_HOST", "0.0.0.0")
PORT = int(os.environ.get("EMBEDDING_PORT", "8000"))

# Request limits. The body limit applies to every endpoint; the text limits apply
# to /embed texts and to the query and text of each /score pair. The models
# truncate to a few hundred tokens anyway, so longer texts only cost memory.
MAX_REQUEST_MB = max(1, int(os.environ.get("EMBEDDING_MAX_REQUEST_MB", "16")))
MAX_TEXTS = max(1, int(os.environ.get("EMBEDDING_MAX_TEXTS", "256")))
MAX_TEXT_CHARS = max(1, int(os.environ.get("EMBEDDING_MAX_TEXT_CHARS", "100000")))
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_MB * 1024 * 1024

# /embed and /score requests served at once. One more is refused with 503 at
# once instead of waiting: below gunicorn's thread count, so /health, /ready and
# the refusals always find a thread (gunicorn.conf.py). Each request in flight
# adds a few hundred MB at its peak, and on a small CPU host more parallel
# requests only share the same cores, hence a low default.
MAX_IN_FLIGHT = max(1, int(os.environ.get("EMBEDDING_MAX_IN_FLIGHT", "2")))
_in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)

# Cross-encoder reranking (POST /score). As with /embed, the client may name the
# model in the request and this is the default. A cross-encoder is another kind
# of model than the embedding one and must be multilingual for multilingual
# content, hence a multilingual MiniLM. Setting CROSS_ENCODER_MODEL to an empty
# string turns the default off: the host then does not rerank and answers 503
# unless the request names a model.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_MODEL, _rerank_revision = split_revision(os.environ.get("CROSS_ENCODER_MODEL", DEFAULT_RERANK_MODEL) or None)
RERANK_MODEL_CACHE = max(1, int(os.environ.get("CROSS_ENCODER_MODEL_CACHE", "1")))
# Pairs are truncated to this many tokens: a long-context reranker would
# otherwise default to its whole window, which is slow on CPU.
RERANK_MAX_LENGTH = max(1, int(os.environ.get("CROSS_ENCODER_MAX_LENGTH", "512")))
# The Laraplate client sends at most 64 pairs per request.
RERANK_MAX_PAIRS = max(1, int(os.environ.get("CROSS_ENCODER_MAX_PAIRS", "64")))


def allowed_models(name, default, default_revision):
    """Models a request may name and the pinned revisions: (names or None for any, {name: revision}).

    Loading a model downloads it from Hugging Face, so a client must not be able
    to name an arbitrary one. "*" in the list turns the check off (development).
    """
    pins = {default: default_revision} if default and default_revision else {}
    names = set()
    for entry in env_list(name):
        model, revision = split_revision(entry)
        names.add(model)
        if revision:
            pins[model] = revision
    if "*" in names:
        return None, pins
    return sorted(names | ({default} if default else set())), pins


EMBEDDING_ALLOWED_MODELS, EMBEDDING_PINS = allowed_models("EMBEDDING_ALLOWED_MODELS", DEFAULT_MODEL, _default_revision)
RERANK_ALLOWED_MODELS, RERANK_PINS = allowed_models("CROSS_ENCODER_ALLOWED_MODELS", RERANK_MODEL, _rerank_revision)


def model_allowed(name, allowed):
    return allowed is None or name in allowed

# Per-family input prefixes. Some embedding models are trained to distinguish a
# search query from an indexed passage and require a textual prefix to embed
# them into the right space; others (MiniLM, mpnet, gte, bge-m3, ...) want none.
#
# The client stays model-agnostic: it always sends the *semantic* intent via
# "input_type" ("query" or "passage"), and the server translates it into the
# right prefix for the resolved model. To support a new family, add an entry
# here — nothing changes client-side.
#
# Matching is by case-insensitive substring on the model name; the first match
# wins, so order more specific keys before more generic ones.
MODEL_PREFIXES = {
    "e5": {"query": "query: ", "passage": "passage: "},
    "nomic": {"query": "search_query: ", "passage": "search_document: "},
    # BGE is intentionally omitted: v1.5 wants an instruction only on queries
    # while bge-m3 wants nothing, so a blanket rule would be wrong. Add a
    # specific entry (e.g. "bge-large") if/when you actually use one.
}

# name -> SentenceTransformer, ordered oldest..newest (simple LRU by re-insert).
_models = {}
# name -> revision commit the model was loaded at, or None when unknown.
_revisions = {}
# Guards both model caches: gunicorn serves requests from several threads.
_models_lock = threading.Lock()


def resolved_revision(name, pinned):
    """The commit of the cached snapshot a model loads from, or None when it cannot be told.

    The Hugging Face cache keeps each revision under snapshots/<commit>/; a local
    path, or a cache without the file, gives None.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(name, "config.json", revision=pinned)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts[:-1] else None


def record_revision(kind, name, pinned):
    _revisions[(kind, name)] = resolved_revision(name, pinned)
    logger.info("Loaded %s model %s at revision %s", kind, name, _revisions[(kind, name)] or "unknown")


def resident_revisions(kind, names):
    return {name: _revisions.get((kind, name)) for name in names}


def load_model(name):
    name = name or DEFAULT_MODEL
    with _models_lock:
        model = _models.pop(name, None)
        if model is None:
            revision = EMBEDDING_PINS.get(name)
            model = SentenceTransformer(name, revision=revision)
            record_revision("embedding", name, revision)
        _models[name] = model  # mark as most-recently used
        while len(_models) > MODEL_CACHE:
            _models.pop(next(iter(_models)))
    return name, model


def prefix_scheme(model_name):
    """Return the {query, passage} prefix dict for a model, or None."""
    lower = model_name.lower()
    for key, scheme in MODEL_PREFIXES.items():
        if key in lower:
            return scheme
    return None


def apply_prefix(texts, model_name, input_type):
    """Ensure each text carries the family prefix for input_type (idempotent).

    The client tells the server the *semantic* intent via input_type; the server
    knows how each model family expects it. If a text already starts with the
    expected prefix (e.g. the client pre-prefixed it, as the Laraplate config
    does), it is kept as-is; otherwise the prefix is prepended. This makes the
    endpoint safe for both "dumb" clients and clients that already prefix, with
    no risk of double prefixes ("query: query: ...").

    No-op (returns texts unchanged) when input_type is omitted or the model has
    no prefix convention.

    Returns (texts, added) where added is True if a prefix was added to at least
    one text.

    Note: detection is a startswith() check, so a passage that literally begins
    with "query: " would be treated as already-prefixed. Acceptable in practice.
    """
    if not input_type:
        return texts, False
    scheme = prefix_scheme(model_name)
    if not scheme or input_type not in scheme:
        return texts, False
    prefix = scheme[input_type]
    added = False
    out = []
    for t in texts:
        if t.startswith(prefix):
            out.append(t)
        else:
            out.append(prefix + t)
            added = True
    return out, added


# name -> CrossEncoder, same LRU scheme as the embedding models.
_rerankers = {}


def load_reranker(name):
    """Return (name, CrossEncoder), or (None, None) when no model is configured."""
    name = name or RERANK_MODEL
    if not name:
        return None, None
    with _models_lock:
        model = _rerankers.pop(name, None)
        if model is None:
            revision = RERANK_PINS.get(name)
            model = CrossEncoder(name, max_length=RERANK_MAX_LENGTH, revision=revision)
            record_revision("rerank", name, revision)
        _rerankers[name] = model  # mark as most-recently used
        while len(_rerankers) > RERANK_MODEL_CACHE:
            _rerankers.pop(next(iter(_rerankers)))
    return name, model


def reranker_info():
    return {
        "default_model": RERANK_MODEL,
        "loaded_models": list(_rerankers.keys()),
        "model_cache": RERANK_MODEL_CACHE,
        "max_length": RERANK_MAX_LENGTH,
        "max_pairs": RERANK_MAX_PAIRS,
    }


def validate_pairs(pairs):
    """Return an error message for a malformed `pairs` value, or None."""
    if not isinstance(pairs, list):
        return 'pairs must be a list of {"query": str, "text": str}'
    if len(pairs) > RERANK_MAX_PAIRS:
        return f"too many pairs: {len(pairs)} (max {RERANK_MAX_PAIRS})"
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, dict)
            or not isinstance(pair.get("query"), str)
            or not isinstance(pair.get("text"), str)
        ):
            return f'pairs[{index}] must be {{"query": str, "text": str}}'
        if len(pair["query"]) > MAX_TEXT_CHARS or len(pair["text"]) > MAX_TEXT_CHARS:
            return f"pairs[{index}] is too long (max {MAX_TEXT_CHARS} characters per query or text)"
    return None


def validate_texts(texts):
    """Return an error message for a malformed or oversized list of texts, or None."""
    if not isinstance(texts, list):
        return "texts must be a list of strings"
    if len(texts) > MAX_TEXTS:
        return f"too many texts: {len(texts)} (max {MAX_TEXTS})"
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            return f"texts[{index}] must be a string"
        if len(text) > MAX_TEXT_CHARS:
            return f"texts[{index}] is too long: {len(text)} characters (max {MAX_TEXT_CHARS})"
    return None


def validate_requested_model(requested, allowed):
    """Return an error message for a `model` the request may not name, or None."""
    if requested is None:
        return None
    if not isinstance(requested, str) or requested == "":
        return "model must be a non-empty string"
    if not model_allowed(requested, allowed):
        return f'model "{requested}" is not allowed on this host'
    return None


def authorized(header):
    """True when the Authorization header carries one of the configured Bearer keys."""
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return False
    return any(hmac.compare_digest(token.encode(), key.encode()) for key in API_KEYS)


def score_pairs(model, pairs):
    """One relevance score in [0, 1] per pair, in the order received.

    The sigmoid is passed explicitly. With activation_fn=None the library falls
    back to the model's own default, which is a sigmoid for some checkpoints and
    raw logits for others (the MS MARCO ones). The Laraplate client clamps
    scores to [0, 1], so raw logits would collapse into ties and lose the order.
    """
    from torch import nn

    raw = model.predict(
        [(pair["query"], pair["text"]) for pair in pairs],
        activation_fn=nn.Sigmoid(),
        convert_to_numpy=True,
    )
    if getattr(raw, "ndim", 1) != 1 or len(raw) != len(pairs):
        raise ValueError(
            "the model does not return one score per pair: it is not a single-score cross-encoder"
        )
    scores = [float(score) for score in raw]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("the model returned a non-finite score")
    return scores


# gunicorn's exit code for a worker that failed to boot: gunicorn stops instead of
# starting the worker again, and systemd or the container runtime restarts it.
LOAD_FAILURE_EXIT_CODE = 3

# Set once the default models are loaded.
ready = threading.Event()


def load_default_models():
    """Warm the defaults so the first request is not a cold load.

    Runs in a background thread so the server answers /health at once and /ready
    says when the models are there. A default model that cannot be loaded ends
    the process instead of leaving it to answer 503 forever.
    """
    try:
        load_model(DEFAULT_MODEL)
        if RERANK_MODEL:
            load_reranker(RERANK_MODEL)
    except Exception:  # noqa: BLE001
        logger.exception("A default model failed to load; stopping the process")
        os._exit(LOAD_FAILURE_EXIT_CODE)
        return
    ready.set()


# Started at import, so in each gunicorn worker: the app must not be preloaded in
# the master (gunicorn.conf.py), whose threads do not survive the fork.
loader = threading.Thread(target=load_default_models, name="model-loader", daemon=True)
loader.start()


def request_id():
    """The id of the current request, made on first use."""
    if "request_id" not in g:
        incoming = request.headers.get("X-Request-Id", "")
        g.request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
    return g.request_id


@app.before_request
def assign_request_id():
    """Registered first, so every later hook and the logs see the id."""
    request_id()


@app.after_request
def add_request_id(response):
    response.headers["X-Request-Id"] = request_id()
    return response


@app.before_request
def require_api_key():
    if not API_KEYS or request.path in PUBLIC_PATHS:
        return None
    if authorized(request.headers.get("Authorization", "")):
        return None
    return jsonify({"error": "unauthorized"}), 401, {"WWW-Authenticate": "Bearer"}


@app.before_request
def require_ready():
    """Runs after the key check, so an unauthorized request still gets 401."""
    if request.path in WORK_PATHS and not ready.is_set():
        return jsonify({"error": "loading models"}), 503, {"Retry-After": "10"}
    return None


@app.before_request
def limit_in_flight():
    """Runs last: a request refused earlier never takes a slot."""
    if request.path not in WORK_PATHS:
        return None
    if not _in_flight.acquire(blocking=False):
        return jsonify({"error": "busy"}), 503, {"Retry-After": "2"}
    g.holds_in_flight_slot = True
    return None


@app.teardown_request
def release_in_flight_slot(error):
    """Runs on every outcome, including an exception or a client that went away."""
    if g.pop("holds_in_flight_slot", False):
        _in_flight.release()


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": f"request body too large (max {MAX_REQUEST_MB} MB)"}), 413


@app.errorhandler(Exception)
def internal_error(error):
    """A generic answer for anything unexpected; the details go to the log only."""
    if isinstance(error, HTTPException):
        return error
    logger.exception("Unhandled error on %s (request %s)", request.path, request_id())
    return jsonify({"error": "internal error", "request_id": request_id()}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "readiness": READINESS,
        "version": VERSION,
        "api_version": API_VERSION,
        "model": DEFAULT_MODEL,
        "default_model": DEFAULT_MODEL,
        "loaded_models": list(_models.keys()),
        "model_cache": MODEL_CACHE,
        "reranker": reranker_info(),
    })


@app.route("/ready", methods=["GET"])
def ready_check():
    """Readiness: 200 once the default models are loaded, 503 until then."""
    if ready.is_set():
        return jsonify({"status": "ready"})
    return jsonify({"status": "loading"}), 503


@app.route("/models", methods=["GET"])
def models():
    """Discovery endpoint: what the client can ask for and how input_type maps.

    It exposes the default, the models a request may name (null when any is
    allowed), what is currently resident and at which revision, and the
    input_type prefix families so the client knows when input_type matters.
    """
    return jsonify({
        "default_model": DEFAULT_MODEL,
        "allowed_models": EMBEDDING_ALLOWED_MODELS,
        "loaded_models": list(_models.keys()),
        "revisions": resident_revisions("embedding", _models.keys()),
        "model_cache": MODEL_CACHE,
        "max_texts": MAX_TEXTS,
        "max_text_chars": MAX_TEXT_CHARS,
        "input_types": ["query", "passage"],
        "prefix_families": MODEL_PREFIXES,
        "reranker": {
            **reranker_info(),
            "allowed_models": RERANK_ALLOWED_MODELS,
            "revisions": resident_revisions("rerank", _rerankers.keys()),
        },
    })


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "the body must be a JSON object"}), 400

    if "texts" in data:
        texts = data["texts"]
    elif "text" in data:
        texts = [data["text"]]
    else:
        return jsonify({"error": "No text or texts provided"}), 400

    error = validate_texts(texts)
    if error:
        return jsonify({"error": error}), 400

    input_type = data.get("input_type")
    if input_type is not None and input_type not in ("query", "passage"):
        return jsonify({
            "error": "input_type must be 'query' or 'passage'",
        }), 400

    requested = data.get("model") or None
    error = validate_requested_model(requested, EMBEDDING_ALLOWED_MODELS)
    if error:
        return jsonify({"error": error}), 400

    name, model = load_model(requested)
    texts, prefix_applied = apply_prefix(texts, name, input_type)
    normalize = bool(data.get("normalize_embeddings", True))
    embeddings = model.encode(texts, normalize_embeddings=normalize)

    return jsonify({
        "model": name,
        "input_type": input_type,
        "prefix_applied": prefix_applied,
        "embeddings": [embedding.tolist() for embedding in embeddings],
    })


@app.route("/score", methods=["POST"])
def score():
    """Relevance score in [0, 1] for each {query, text} pair, from a cross-encoder."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "the body must be a JSON object"}), 400
    pairs = data.get("pairs")

    error = validate_pairs(pairs)
    if error:
        return jsonify({"error": error}), 400

    requested = data.get("model")
    error = validate_requested_model(requested, RERANK_ALLOWED_MODELS)
    if error:
        return jsonify({"error": error}), 400

    name, model = load_reranker(requested)
    if model is None:
        return jsonify({
            "error": 'No cross-encoder model configured: set CROSS_ENCODER_MODEL or send "model"',
        }), 503

    return jsonify({
        "model": name,
        "scores": score_pairs(model, pairs) if pairs else [],
    })


if __name__ == "__main__":
    # Development server only; production runs under gunicorn (gunicorn.conf.py).
    app.run(host=HOST, port=PORT)
