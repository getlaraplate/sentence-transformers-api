"""Tests for the Bearer-key check, the model allow-lists and the request limits.

    python -m unittest discover -s tests
"""

import unittest

from service_case import FakeCrossEncoder, FakeSentenceTransformer, ServiceTestCase, pair

KEY = "s3cret-key"


class AuthenticationTest(ServiceTestCase):
    def protected_requests(self, client, headers):
        return {
            "/embed": client.post("/embed", json={"texts": ["ciao"]}, headers=headers),
            "/score": client.post("/score", json={"pairs": [pair()]}, headers=headers),
            "/models": client.get("/models", headers=headers),
        }

    def test_without_a_key_configured_every_endpoint_is_open(self):
        client = self.service()

        for path, response in self.protected_requests(client, {}).items():
            with self.subTest(path):
                self.assertEqual(response.status_code, 200)

    def test_with_a_key_configured_requests_without_it_answer_401(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY)

        for path, response in self.protected_requests(client, {}).items():
            with self.subTest(path):
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.get_json(), {"error": "unauthorized"})
                self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

    def test_a_wrong_key_or_another_scheme_answers_401(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY)
        headers = {
            "wrong key": f"Bearer {KEY}x",
            "basic scheme": f"Basic {KEY}",
            "bare key": KEY,
            "empty bearer": "Bearer ",
        }

        for label, header in headers.items():
            with self.subTest(label):
                response = client.post("/embed", json={"texts": ["ciao"]}, headers={"Authorization": header})

                self.assertEqual(response.status_code, 401)

    def test_the_bearer_header_the_laraplate_client_sends_is_accepted(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY)

        for path, response in self.protected_requests(client, {"Authorization": f"Bearer {KEY}"}).items():
            with self.subTest(path):
                self.assertEqual(response.status_code, 200)

    def test_several_comma_separated_keys_are_each_accepted(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=f"{KEY}, rerank-key")

        for key in (KEY, "rerank-key"):
            with self.subTest(key):
                response = client.post("/score", json={"pairs": [pair()]}, headers={"Authorization": f"Bearer {key}"})

                self.assertEqual(response.status_code, 200)

    def test_health_stays_open_when_a_key_is_configured(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY)

        self.assertEqual(client.get("/health").status_code, 200)

    def test_an_unauthorized_request_does_not_reach_the_model(self):
        client = self.service(SENTENCE_TRANSFORMERS_API_KEY=KEY, CROSS_ENCODER_ALLOWED_MODELS="*")

        client.post("/score", json={"pairs": [pair()], "model": "org/other"})

        self.assertEqual([model.name for model in FakeCrossEncoder.instances], ["cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"])


class AllowedModelsTest(ServiceTestCase):
    def test_embed_accepts_the_default_model_by_name_or_by_omission(self):
        client = self.service(EMBEDDING_MODEL="org/embedder")

        for body in ({"texts": ["ciao"]}, {"texts": ["ciao"], "model": "org/embedder"}):
            with self.subTest(body):
                self.assertEqual(client.post("/embed", json=body).status_code, 200)

    def test_embed_rejects_a_model_that_is_not_allowed_without_loading_it(self):
        client = self.service(EMBEDDING_MODEL="org/embedder")

        response = client.post("/embed", json={"texts": ["ciao"], "model": "evil/model"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("not allowed", response.get_json()["error"])
        self.assertEqual([model.name for model in FakeSentenceTransformer.instances], ["org/embedder"])

    def test_embed_accepts_the_models_listed_in_addition_to_the_default(self):
        client = self.service(EMBEDDING_MODEL="org/embedder", EMBEDDING_ALLOWED_MODELS="org/second, org/third")

        for model in ("org/embedder", "org/second", "org/third"):
            with self.subTest(model):
                response = client.post("/embed", json={"texts": ["ciao"], "model": model})

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["model"], model)

    def test_embed_rejects_a_model_that_is_not_a_string(self):
        client = self.service()

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"], "model": 5}).status_code, 400)

    def test_score_rejects_a_model_that_is_not_allowed_without_loading_it(self):
        client = self.service(CROSS_ENCODER_MODEL="org/reranker")

        response = client.post("/score", json={"pairs": [pair()], "model": "evil/model"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("not allowed", response.get_json()["error"])
        self.assertEqual([model.name for model in FakeCrossEncoder.instances], ["org/reranker"])

    def test_a_star_allows_any_model(self):
        client = self.service(EMBEDDING_ALLOWED_MODELS="*", CROSS_ENCODER_ALLOWED_MODELS="*")

        self.assertEqual(client.post("/embed", json={"texts": ["ciao"], "model": "org/any"}).status_code, 200)
        self.assertEqual(client.post("/score", json={"pairs": [pair()], "model": "org/any"}).status_code, 200)

    def test_models_lists_what_a_request_may_name(self):
        client = self.service(
            EMBEDDING_MODEL="org/embedder",
            EMBEDDING_ALLOWED_MODELS="org/second",
            CROSS_ENCODER_MODEL="org/reranker",
            CROSS_ENCODER_ALLOWED_MODELS="*",
        )

        models = client.get("/models").get_json()

        self.assertEqual(models["allowed_models"], ["org/embedder", "org/second"])
        self.assertIsNone(models["reranker"]["allowed_models"])


class RequestLimitsTest(ServiceTestCase):
    def test_embed_rejects_more_texts_than_the_limit(self):
        client = self.service(EMBEDDING_MAX_TEXTS="2")

        self.assertEqual(client.post("/embed", json={"texts": ["a", "b"]}).status_code, 200)
        response = client.post("/embed", json={"texts": ["a", "b", "c"]})

        self.assertEqual(response.status_code, 400)
        self.assertIn("too many texts", response.get_json()["error"])

    def test_embed_rejects_a_text_longer_than_the_limit(self):
        client = self.service(EMBEDDING_MAX_TEXT_CHARS="5")

        for body in ({"texts": ["ok", "too long"]}, {"text": "too long"}):
            with self.subTest(body):
                response = client.post("/embed", json=body)

                self.assertEqual(response.status_code, 400)
                self.assertIn("too long", response.get_json()["error"])

    def test_embed_rejects_texts_that_are_not_strings(self):
        client = self.service()
        bodies = {
            "texts is not a list": {"texts": "ciao"},
            "an item is not a string": {"texts": ["ciao", 5]},
            "text is not a string": {"text": 5},
            "the body is not an object": ["ciao"],
        }

        for label, body in bodies.items():
            with self.subTest(label):
                self.assertEqual(client.post("/embed", json=body).status_code, 400)

    def test_score_rejects_a_pair_longer_than_the_limit(self):
        client = self.service(EMBEDDING_MAX_TEXT_CHARS="5")

        for item in (pair("q", "too long"), pair("too long", "t")):
            with self.subTest(item):
                response = client.post("/score", json={"pairs": [item]})

                self.assertEqual(response.status_code, 400)
                self.assertIn("too long", response.get_json()["error"])

    def test_a_body_larger_than_the_limit_answers_413_in_json(self):
        client = self.service(EMBEDDING_MAX_REQUEST_MB="1")
        oversized = {"texts": ["a" * 600_000, "b" * 600_000]}

        for path in ("/embed", "/score"):
            with self.subTest(path):
                response = client.post(path, json=oversized)

                self.assertEqual(response.status_code, 413)
                self.assertIn("too large", response.get_json()["error"])

    def test_the_default_limits_accept_what_the_laraplate_client_sends(self):
        client = self.service()

        response = client.post("/embed", json={"texts": ["passage: " + "a" * 4000] * 128})

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
