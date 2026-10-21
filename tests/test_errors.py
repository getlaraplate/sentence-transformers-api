"""Tests for X-Request-Id and for the generic answer of unexpected errors.

    python -m unittest discover -s tests
"""

import re
import unittest

import numpy

from service_case import FakeCrossEncoder, FakeSentenceTransformer, ServiceTestCase, pair

KEY = "s3cret-key"
GENERATED_ID = re.compile(r"^[0-9a-f]{32}$")


class RequestIdTest(ServiceTestCase):
    def test_a_valid_incoming_id_is_echoed(self):
        client = self.service()

        response = client.post("/embed", json={"texts": ["ciao"]}, headers={"X-Request-Id": "laraplate.job-42_A"})

        self.assertEqual(response.headers["X-Request-Id"], "laraplate.job-42_A")

    def test_an_invalid_incoming_id_is_replaced(self):
        client = self.service()

        for label, value in {"too long": "a" * 129, "a space": "a b", "a tab": "a\tb", "a slash": "a/b"}.items():
            with self.subTest(label):
                response = client.get("/health", headers={"X-Request-Id": value})

                self.assertRegex(response.headers["X-Request-Id"], GENERATED_ID)

    def test_every_answer_carries_an_id(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY, EMBEDDING_MAX_REQUEST_MB="1")
        auth = {"Authorization": f"Bearer {KEY}"}
        answers = {
            "200": client.get("/health"),
            "400": client.post("/embed", json={"texts": 5}, headers=auth),
            "401": client.post("/embed", json={"texts": ["ciao"]}),
            "404": client.get("/nowhere", headers=auth),
            "413": client.post("/embed", json={"texts": ["a" * 1_100_000]}, headers=auth),
        }

        for label, response in answers.items():
            with self.subTest(label):
                self.assertEqual(str(response.status_code), label)
                self.assertRegex(response.headers["X-Request-Id"], GENERATED_ID)


class GenericErrorTest(ServiceTestCase):
    def test_an_unexpected_error_answers_a_generic_500_and_logs_the_exception(self):
        client = self.service()
        FakeSentenceTransformer.encode_fails_with = RuntimeError("secret path /srv/models")

        with self.assertLogs("sentence_api", "ERROR") as logs:
            response = client.post("/embed", json={"texts": ["ciao"]}, headers={"X-Request-Id": "req-1"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "internal error", "request_id": "req-1"})
        self.assertNotIn("secret", response.get_data(as_text=True))
        self.assertIn("req-1", logs.output[0])
        self.assertIn("secret path", "\n".join(logs.output))

    def test_a_cross_encoder_that_is_not_single_score_answers_a_generic_500(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")
        FakeCrossEncoder.scores = numpy.array([[0.1, 0.9]])

        with self.assertLogs("sentence_api", "ERROR"):
            response = client.post("/score", json={"pairs": [pair()]})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "internal error")
        self.assertNotIn("cross-encoder", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
