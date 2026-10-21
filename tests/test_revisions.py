"""Tests for the reported model revisions and for pinning one with name@revision.

    python -m unittest discover -s tests
"""

import sys
import types
import unittest
from unittest import mock

from service_case import FakeCrossEncoder, FakeSentenceTransformer, ServiceTestCase, pair


def fake_hub(snapshots):
    """A huggingface_hub stand-in whose cache knows the given {(name, revision): commit}."""
    hub = types.ModuleType("huggingface_hub")
    hub.calls = []

    def try_to_load_from_cache(repo_id, filename, revision=None, **kwargs):
        hub.calls.append((repo_id, filename, revision))
        commit = snapshots.get((repo_id, revision))
        if commit is None:
            return None
        return f"/cache/hub/models--{repo_id.replace('/', '--')}/snapshots/{commit}/{filename}"

    hub.try_to_load_from_cache = try_to_load_from_cache
    return hub


class RevisionTest(ServiceTestCase):
    def with_hub(self, snapshots):
        hub = fake_hub(snapshots)
        patcher = mock.patch.dict(sys.modules, {"huggingface_hub": hub})
        patcher.start()
        self.addCleanup(patcher.stop)
        return hub

    def test_models_reports_the_revision_of_each_resident_model(self):
        self.with_hub({("org/embedder", None): "aaa111", ("org/reranker", None): "bbb222"})
        client = self.service(EMBEDDING_MODEL="org/embedder", CROSS_ENCODER_MODEL="org/reranker")

        models = client.get("/models").get_json()

        self.assertEqual(models["revisions"], {"org/embedder": "aaa111"})
        self.assertEqual(models["reranker"]["revisions"], {"org/reranker": "bbb222"})

    def test_an_unknown_revision_is_reported_as_null(self):
        self.with_hub({})
        client = self.service(EMBEDDING_MODEL="org/embedder")

        self.assertEqual(client.get("/models").get_json()["revisions"], {"org/embedder": None})

    def test_without_the_hub_library_the_revision_is_null(self):
        patcher = mock.patch.dict(sys.modules, {"huggingface_hub": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        client = self.service(EMBEDDING_MODEL="org/embedder")

        self.assertEqual(client.get("/models").get_json()["revisions"], {"org/embedder": None})

    def test_the_loaded_revision_is_logged(self):
        self.with_hub({("org/embedder", None): "aaa111"})

        with self.assertLogs("sentence_api", "INFO") as logs:
            self.service(EMBEDDING_MODEL="org/embedder", CROSS_ENCODER_MODEL="")

        self.assertTrue(any("org/embedder" in line and "aaa111" in line for line in logs.output))

    def test_a_pinned_default_loads_that_revision_and_is_named_without_it(self):
        hub = self.with_hub({("org/embedder", "abc"): "abc"})
        client = self.service(EMBEDDING_MODEL="org/embedder@abc")

        response = client.post("/embed", json={"texts": ["ciao"], "model": "org/embedder"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model"], "org/embedder")
        self.assertEqual(FakeSentenceTransformer.instances[0].name, "org/embedder")
        self.assertEqual(FakeSentenceTransformer.instances[0].options["revision"], "abc")
        self.assertIn(("org/embedder", "config.json", "abc"), hub.calls)
        self.assertEqual(client.get("/health").get_json()["default_model"], "org/embedder")

    def test_a_request_may_not_name_the_revision(self):
        self.with_hub({})
        client = self.service(EMBEDDING_MODEL="org/embedder@abc")

        response = client.post("/embed", json={"texts": ["ciao"], "model": "org/embedder@abc"})

        self.assertEqual(response.status_code, 400)

    def test_an_allowed_model_may_be_pinned(self):
        self.with_hub({})
        client = self.service(EMBEDDING_ALLOWED_MODELS="org/second@def")

        response = client.post("/embed", json={"texts": ["ciao"], "model": "org/second"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(FakeSentenceTransformer.instances[-1].options["revision"], "def")
        self.assertIn("org/second", client.get("/models").get_json()["allowed_models"])

    def test_unpinned_models_load_the_latest_revision(self):
        self.with_hub({})
        self.service(EMBEDDING_MODEL="org/embedder")

        self.assertIsNone(FakeSentenceTransformer.instances[0].options.get("revision"))

    def test_a_pinned_reranker_loads_that_revision(self):
        self.with_hub({})
        client = self.service(CROSS_ENCODER_MODEL="org/reranker@ghi", CROSS_ENCODER_ALLOWED_MODELS="org/other@jkl")

        client.post("/score", json={"pairs": [pair()], "model": "org/other"})

        revisions = {model.name: model.options.get("revision") for model in FakeCrossEncoder.instances}
        self.assertEqual(revisions, {"org/reranker": "ghi", "org/other": "jkl"})
        self.assertEqual(client.get("/health").get_json()["reranker"]["default_model"], "org/reranker")


if __name__ == "__main__":
    unittest.main()
