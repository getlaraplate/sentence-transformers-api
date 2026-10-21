"""Tests that the service and openapi.json describe the same HTTP contract, and that /health declares it.

    python -m unittest discover -s tests
"""

import threading
import unittest
from pathlib import Path

import numpy

from contract import DOCUMENT, problems, resolve, response_schema
from service_case import FakeCrossEncoder, FakeSentenceTransformer, ServiceTestCase, pair

KEY = "s3cret-key"
VERSION = (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()


class VersionTest(ServiceTestCase):
    def test_health_declares_the_release_and_the_api_version(self):
        health = self.service().get("/health").get_json()

        self.assertEqual((health["version"], health["api_version"]), (VERSION, 1))

    def test_the_document_carries_the_same_versions(self):
        self.assertEqual((DOCUMENT["info"]["version"], DOCUMENT["info"]["x-api-version"]), (VERSION, 1))


class ContractTest(ServiceTestCase):
    def assertMatches(self, path, method, response):
        schema = response_schema(path, method, response.status_code)
        self.assertEqual(problems(schema, response.get_json()), [], f"{method.upper()} {path} {response.status_code}")
        self.assertIn("X-Request-Id", response.headers)

    def test_the_routes_and_the_documented_paths_match(self):
        module = self.load_service()
        routes = {
            (rule.rule, method.lower())
            for rule in module.app.url_map.iter_rules()
            if rule.endpoint != "static"
            for method in rule.methods - {"HEAD", "OPTIONS"}
        }
        documented = {(path, method) for path, operations in DOCUMENT["paths"].items() for method in operations}

        self.assertEqual(routes, documented)

    def test_every_example_matches_its_schema(self):
        for path, operations in DOCUMENT["paths"].items():
            for method, operation in operations.items():
                body = operation.get("requestBody", {}).get("content", {}).get("application/json")
                if body:
                    with self.subTest(f"{method} {path} request"):
                        self.assertEqual(problems(body["schema"], body["example"]), [])
                for status, answer in operation["responses"].items():
                    with self.subTest(f"{method} {path} {status}"):
                        content = answer["content"]["application/json"]
                        self.assertEqual(problems(content["schema"], content["example"]), [])

    def test_the_answers_match_the_document(self):
        client = self.service(
            SENTENCE_TRANSFORMERS_API_KEY=KEY,
            EMBEDDING_MAX_REQUEST_MB="1",
            CROSS_ENCODER_MODEL="org/reranker",
        )
        auth = {"Authorization": f"Bearer {KEY}"}
        too_large = {"texts": ["a" * 1_100_000]}

        self.assertMatches("/health", "get", client.get("/health"))
        self.assertMatches("/ready", "get", client.get("/ready"))
        self.assertMatches("/models", "get", client.get("/models", headers=auth))
        self.assertMatches("/models", "get", client.get("/models"))
        self.assertMatches("/embed", "post", client.post("/embed", json={"texts": ["ciao"]}, headers=auth))
        self.assertMatches("/embed", "post", client.post("/embed", json={"texts": 5}, headers=auth))
        self.assertMatches("/embed", "post", client.post("/embed", json={"texts": ["ciao"]}))
        self.assertMatches("/embed", "post", client.post("/embed", json=too_large, headers=auth))
        self.assertMatches("/score", "post", client.post("/score", json={"pairs": [pair()]}, headers=auth))
        self.assertMatches("/score", "post", client.post("/score", json={"pairs": "x"}, headers=auth))
        self.assertMatches("/score", "post", client.post("/score", json=too_large, headers=auth))

        FakeSentenceTransformer.encode_fails_with = RuntimeError("model crashed")
        with self.assertLogs("sentence_api", "ERROR"):
            self.assertMatches("/embed", "post", client.post("/embed", json={"texts": ["ciao"]}, headers=auth))
        FakeCrossEncoder.scores = numpy.array([[0.1, 0.9]])
        with self.assertLogs("sentence_api", "ERROR"):
            self.assertMatches("/score", "post", client.post("/score", json={"pairs": [pair()]}, headers=auth))

    def test_the_503_answers_match_the_document(self):
        loaded = threading.Event()
        FakeSentenceTransformer.block_on = loaded
        module = self.load_service(wait_until_ready=False)
        self.addCleanup(module.loader.join, 5)
        self.addCleanup(loaded.set)
        client = module.app.test_client()

        self.assertMatches("/ready", "get", client.get("/ready"))
        self.assertMatches("/embed", "post", client.post("/embed", json={"texts": ["ciao"]}))
        self.assertMatches("/score", "post", client.post("/score", json={"pairs": [pair()]}))

        FakeSentenceTransformer.block_on = None
        no_reranker = self.service(CROSS_ENCODER_MODEL="")
        self.assertMatches("/score", "post", no_reranker.post("/score", json={"pairs": [pair()]}))

    def test_the_checker_catches_a_missing_field_and_a_wrong_type(self):
        schema = resolve({"$ref": "#/components/schemas/ScoreResponse"})

        self.assertEqual(problems(schema, {"model": "m"}), ["$: missing scores"])
        self.assertEqual(problems(schema, {"model": "m", "scores": ["x"]}), ["$.scores[0]: expected number, got str"])


if __name__ == "__main__":
    unittest.main()
