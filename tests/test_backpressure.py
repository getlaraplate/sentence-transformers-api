"""Tests for the in-flight limit on /embed and /score.

    python -m unittest discover -s tests
"""

import threading
import unittest

from service_case import FakeSentenceTransformer, ServiceTestCase


class InFlightLimitTest(ServiceTestCase):
    def occupy_the_only_slot(self, client):
        """Start an /embed that holds the single slot until the returned event is set."""
        FakeSentenceTransformer.encoding = threading.Event()
        FakeSentenceTransformer.encode_release = threading.Event()
        slow = threading.Thread(target=client.post, args=("/embed",), kwargs={"json": {"texts": ["slow"]}})
        slow.start()
        self.addCleanup(slow.join, 5)
        self.addCleanup(FakeSentenceTransformer.encode_release.set)
        FakeSentenceTransformer.encoding.wait(5)
        return FakeSentenceTransformer.encode_release, slow

    def test_a_request_beyond_the_limit_answers_503_at_once(self):
        client = self.service(EMBEDDING_MAX_IN_FLIGHT="1")
        self.occupy_the_only_slot(client)

        for path, body in (("/embed", {"texts": ["ciao"]}), ("/score", {"pairs": []})):
            with self.subTest(path):
                response = client.post(path, json=body)

                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.get_json(), {"error": "busy"})
                self.assertEqual(response.headers["Retry-After"], "2")

    def test_the_slot_comes_back_when_the_request_ends(self):
        client = self.service(EMBEDDING_MAX_IN_FLIGHT="1")
        release, slow = self.occupy_the_only_slot(client)

        release.set()
        slow.join(5)
        FakeSentenceTransformer.encode_release = None

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"]}).status_code, 200)

    def test_the_slot_comes_back_after_a_failure(self):
        client = self.service(EMBEDDING_MAX_IN_FLIGHT="1")
        FakeSentenceTransformer.encode_fails_with = RuntimeError("model crashed")

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"]}).status_code, 500)
        FakeSentenceTransformer.encode_fails_with = None

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"]}).status_code, 200)

    def test_the_slot_comes_back_after_a_refused_body(self):
        client = self.service(EMBEDDING_MAX_IN_FLIGHT="1")

        self.assertEqual(client.post("/embed", json={"texts": "not a list"}).status_code, 400)

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"]}).status_code, 200)

    def test_health_ready_and_models_are_not_counted(self):
        client = self.service(EMBEDDING_MAX_IN_FLIGHT="1")
        self.occupy_the_only_slot(client)

        for path in ("/health", "/ready", "/models"):
            with self.subTest(path):
                self.assertEqual(client.get(path).status_code, 200)

    def test_the_limit_defaults_to_six(self):
        module = self.load_service()

        self.assertEqual(module.MAX_IN_FLIGHT, 6)


if __name__ == "__main__":
    unittest.main()
