"""Tests for POST /score (cross-encoder reranking) and the reranker block of /health and /models.

sentence-transformers and torch are replaced by small fakes: the tests are about what the service
asks of a model and what it answers, not about a real checkpoint, and they run without downloading one.

    python -m unittest discover -s tests
"""

import unittest

import numpy

from service_case import FakeCrossEncoder, FakeSigmoid, ServiceTestCase, pair


class ScoreEndpointTest(ServiceTestCase):
    def test_scores_each_pair_in_order_and_names_the_model(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        response = client.post("/score", json={"pairs": [pair(text="a" * 10), pair(text="a" * 40)]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"model": "org/reranker", "scores": [0.1, 0.4]})

    def test_sends_the_pairs_to_the_model_as_query_text_tuples(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        client.post("/score", json={"pairs": [pair("q1", "t1"), pair("q2", "t2")]})

        self.assertEqual(FakeCrossEncoder.instances[0].calls[0]["pairs"], [("q1", "t1"), ("q2", "t2")])

    def test_asks_for_a_sigmoid_explicitly_because_the_model_default_may_be_raw_logits(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        client.post("/score", json={"pairs": [pair()]})

        self.assertIsInstance(FakeCrossEncoder.instances[0].calls[0]["activation_fn"], FakeSigmoid)

    def test_uses_the_multilingual_default_model_when_none_is_configured(self):
        client = self.service()

        response = client.post("/score", json={"pairs": [pair()]})

        self.assertEqual(response.get_json()["model"], "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.assertEqual(
            [model.name for model in FakeCrossEncoder.instances],
            ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"],
        )

    def test_loads_the_default_model_at_startup_with_the_length_limit(self):
        self.service(CROSS_ENCODER_MODEL="org/reranker", CROSS_ENCODER_MAX_LENGTH="256")

        self.assertEqual([(m.name, m.max_length) for m in FakeCrossEncoder.instances], [("org/reranker", 256)])

    def test_the_length_limit_defaults_to_512(self):
        self.service(CROSS_ENCODER_MODEL="org/reranker")

        self.assertEqual(FakeCrossEncoder.instances[0].max_length, 512)

    def test_a_request_can_name_another_model_and_the_cache_keeps_only_the_newest(self):
        client = self.service(
            CROSS_ENCODER_MODEL="org/first",
            CROSS_ENCODER_MODEL_CACHE="1",
            CROSS_ENCODER_ALLOWED_MODELS="org/second",
        )

        response = client.post("/score", json={"pairs": [pair()], "model": "org/second"})

        self.assertEqual(response.get_json()["model"], "org/second")
        self.assertEqual(client.get("/health").get_json()["reranker"]["loaded_models"], ["org/second"])

    def test_answers_503_when_the_default_is_turned_off_and_no_model_is_requested(self):
        client = self.service(CROSS_ENCODER_MODEL="")

        response = client.post("/score", json={"pairs": [pair()]})

        self.assertEqual(response.status_code, 503)
        self.assertIn("CROSS_ENCODER_MODEL", response.get_json()["error"])
        self.assertEqual(FakeCrossEncoder.instances, [])

    def test_a_host_with_the_default_off_still_scores_when_the_request_names_an_allowed_model(self):
        client = self.service(CROSS_ENCODER_MODEL="", CROSS_ENCODER_ALLOWED_MODELS="org/reranker")

        response = client.post("/score", json={"pairs": [pair()], "model": "org/reranker"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model"], "org/reranker")

    def test_rejects_malformed_pairs(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")
        bodies = {
            "no pairs": {},
            "pairs is not a list": {"pairs": "a question"},
            "an item is not an object": {"pairs": ["a question"]},
            "an item has no text": {"pairs": [{"query": "q"}]},
            "the query is not a string": {"pairs": [{"query": 1, "text": "t"}]},
        }

        for label, body in bodies.items():
            with self.subTest(label):
                response = client.post("/score", json=body)

                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_rejects_a_body_that_is_not_json(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        response = client.post("/score", data="nope", content_type="text/plain")

        self.assertEqual(response.status_code, 400)

    def test_rejects_more_pairs_than_the_limit(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker", CROSS_ENCODER_MAX_PAIRS="2")

        response = client.post("/score", json={"pairs": [pair(), pair(), pair()]})

        self.assertEqual(response.status_code, 400)
        self.assertIn("too many pairs", response.get_json()["error"])

    def test_the_pair_limit_defaults_to_the_64_the_laraplate_client_sends(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        self.assertEqual(client.post("/score", json={"pairs": [pair()] * 64}).status_code, 200)
        self.assertEqual(client.post("/score", json={"pairs": [pair()] * 65}).status_code, 400)

    def test_rejects_a_model_that_is_not_a_non_empty_string(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        for model in (5, ""):
            with self.subTest(model):
                self.assertEqual(client.post("/score", json={"pairs": [pair()], "model": model}).status_code, 400)

    def test_empty_pairs_answer_an_empty_list_without_calling_the_model(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        response = client.post("/score", json={"pairs": []})

        self.assertEqual(response.get_json(), {"model": "org/reranker", "scores": []})
        self.assertEqual(FakeCrossEncoder.instances[0].calls, [])

    def test_fails_instead_of_answering_when_the_model_does_not_return_one_score_per_pair(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")
        answers = {
            "fewer scores than pairs": numpy.array([0.5]),
            "one row of scores per pair (not a single-score model)": numpy.array([[0.1, 0.9], [0.2, 0.8]]),
            "a score that is not finite": numpy.array([0.5, float("nan")]),
        }

        for label, answer in answers.items():
            with self.subTest(label):
                FakeCrossEncoder.scores = answer

                response = client.post("/score", json={"pairs": [pair(), pair()]})

                self.assertEqual(response.status_code, 500)
                self.assertIn("error", response.get_json())

    def test_reports_whether_it_reranks_in_health_and_models(self):
        with_model = self.service(CROSS_ENCODER_MODEL="org/reranker")
        without_model = self.service(CROSS_ENCODER_MODEL="")

        for client in (with_model, without_model):
            for path in ("/health", "/models"):
                self.assertIn("reranker", client.get(path).get_json())

        self.assertEqual(with_model.get("/health").get_json()["reranker"]["default_model"], "org/reranker")
        self.assertIsNone(without_model.get("/health").get_json()["reranker"]["default_model"])

    def test_health_keeps_the_keys_the_laraplate_client_reads(self):
        client = self.service(EMBEDDING_MODEL="org/embedder")

        health = client.get("/health").get_json()

        self.assertEqual((health["status"], health["model"]), ("healthy", "org/embedder"))

    def test_embed_still_answers(self):
        client = self.service()

        response = client.post("/embed", json={"texts": ["ciao", "hello"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["embeddings"]), 2)


if __name__ == "__main__":
    unittest.main()
