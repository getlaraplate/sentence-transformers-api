"""Shared fakes and base test case: a fresh copy of the service under a given environment.

sentence-transformers and torch are replaced by small fakes: the tests are about what the service
asks of a model and what it answers, not about a real checkpoint, and they run without downloading one.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy

SERVICE = Path(__file__).resolve().parent.parent / "sentence-api.py"


class FakeSentenceTransformer:
    instances = []
    # When set, the constructor waits on this event (a model still loading) or raises this exception.
    block_on = None
    fail_with = None

    def __init__(self, name, **kwargs):
        if type(self).block_on is not None:
            type(self).block_on.wait(5)
        if type(self).fail_with is not None:
            raise type(self).fail_with
        self.name = name
        self.options = kwargs
        type(self).instances.append(self)

    # When set, encode() signals `encoding`, then waits on `encode_release` (a slow request) or raises `encode_fails_with`.
    encoding = None
    encode_release = None
    encode_fails_with = None

    def encode(self, texts, normalize_embeddings=True):
        cls = type(self)
        if cls.encoding is not None:
            cls.encoding.set()
        if cls.encode_release is not None:
            cls.encode_release.wait(5)
        if cls.encode_fails_with is not None:
            raise cls.encode_fails_with
        return numpy.zeros((len(texts), 3))


class FakeSigmoid:
    """Stands for torch.nn.Sigmoid: the test only needs to tell it from "no activation"."""


class FakeCrossEncoder:
    instances = []
    # When set, predict() returns this instead of one score per pair.
    scores = None

    def __init__(self, name, max_length=None, **kwargs):
        self.name = name
        self.max_length = max_length
        self.options = kwargs
        self.calls = []
        type(self).instances.append(self)

    def predict(self, pairs, activation_fn=None, convert_to_numpy=True, **kwargs):
        self.calls.append({"pairs": pairs, "activation_fn": activation_fn})
        if type(self).scores is not None:
            return type(self).scores
        return numpy.array([min(len(text) / 100, 1.0) for _, text in pairs])


def pair(query="how many people live in Berlin", text="Berlin has about 3.5 million inhabitants"):
    return {"query": query, "text": text}


class ServiceTestCase(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.instances = []
        FakeSentenceTransformer.block_on = None
        FakeSentenceTransformer.fail_with = None
        FakeSentenceTransformer.encoding = None
        FakeSentenceTransformer.encode_release = None
        FakeSentenceTransformer.encode_fails_with = None
        FakeCrossEncoder.instances = []
        FakeCrossEncoder.scores = None

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = FakeSentenceTransformer
        sentence_transformers.CrossEncoder = FakeCrossEncoder
        torch = types.ModuleType("torch")
        torch.nn = types.SimpleNamespace(Sigmoid=FakeSigmoid)

        # Kept for the whole test: /score imports torch when it is called, not when the module loads.
        patcher = mock.patch.dict(sys.modules, {"sentence_transformers": sentence_transformers, "torch": torch})
        patcher.start()
        self.addCleanup(patcher.stop)

    def service(self, wait_until_ready=True, **env):
        """A fresh copy of the service, so caches and defaults start from the given environment only.

        The default models load in a background thread; unless told otherwise, wait until it is done.
        """
        return self.load_service(wait_until_ready, **env).app.test_client()

    def load_service(self, wait_until_ready=True, **env):
        """The module of a fresh copy of the service, for tests that need its internals."""
        environ = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("CROSS_ENCODER_", "EMBEDDING_", "SENTENCE_TRANSFORMERS_"))
        }
        environ.update(env)

        spec = importlib.util.spec_from_file_location("sentence_api_under_test", SERVICE)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, environ, clear=True):
            spec.loader.exec_module(module)

        if wait_until_ready:
            module.loader.join(5)

        return module
