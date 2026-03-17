import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.services import llm_router


class TestLlmRouterUnit(unittest.TestCase):
    def setUp(self):
        llm_router._prompt_cache.clear()

    def test_classify_task_coding_math_creative_and_default(self):
        self.assertEqual(llm_router.classify_task("Write a Python function to sort an array"), "coding")
        self.assertEqual(llm_router.classify_task("Solve this integral and show steps"), "math")
        self.assertEqual(llm_router.classify_task("Write a short fiction story with a plot twist"), "creative")
        self.assertEqual(llm_router.classify_task("Capital of France?"), "factual")

    def test_analyze_complexity_ranges(self):
        simple = llm_router.analyze_complexity("Capital of France?")
        self.assertGreaterEqual(simple, 0.0)
        self.assertLessEqual(simple, 0.1)

        complex_prompt = (
            "Analyze and design a distributed microservice architecture using docker and kubernetes, "
            "compare algorithm complexity, explain step by step, and include equations like a+b=c. "
            "Also discuss concurrency and recursion in detail."
        )
        complex_score = llm_router.analyze_complexity(complex_prompt)
        self.assertGreaterEqual(complex_score, 0.25)
        self.assertLessEqual(complex_score, 1.0)

    def test_select_model_thresholds(self):
        self.assertEqual(llm_router.select_model(0.24), llm_router.FAST_MODEL)
        self.assertEqual(llm_router.select_model(0.25), llm_router.MEDIUM_MODEL)
        self.assertEqual(llm_router.select_model(0.55), llm_router.HEAVY_MODEL)

    @patch("app.services.llm_router.generate_completion")
    def test_route_request_uses_cache(self, mock_generate):
        mock_generate.return_value = {
            "response": "First answer with enough words to avoid retry in this test path.",
            "model": llm_router.FAST_MODEL,
            "latency_sec": 0.12,
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22,
        }

        prompt = "Explain HTTP status codes."
        first = llm_router.route_request(prompt)
        second = llm_router.route_request(prompt)

        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(mock_generate.call_count, 1)
        self.assertEqual(first["response"], second["response"])

    @patch("app.services.llm_router.generate_completion")
    def test_route_request_retries_when_fast_response_is_thin(self, mock_generate):
        mock_generate.side_effect = [
            {
                "response": "Too short.",
                "model": llm_router.FAST_MODEL,
                "latency_sec": 0.10,
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
            {
                "response": "This is a longer heavy model response that should pass the quality gate check.",
                "model": llm_router.HEAVY_MODEL,
                "latency_sec": 0.40,
                "prompt_tokens": 5,
                "completion_tokens": 18,
                "total_tokens": 23,
            },
        ]

        result = llm_router.route_request("hello")

        self.assertTrue(result["retried"])
        self.assertEqual(result["model"], llm_router.HEAVY_MODEL)
        self.assertFalse(result["from_cache"])
        self.assertEqual(mock_generate.call_count, 2)
        self.assertEqual(mock_generate.call_args_list[0].args[1], llm_router.FAST_MODEL)
        self.assertEqual(mock_generate.call_args_list[1].args[1], llm_router.HEAVY_MODEL)

    @patch("app.services.llm_router.generate_completion")
    def test_route_request_does_not_retry_when_initial_model_is_not_fast(self, mock_generate):
        mock_generate.return_value = {
            "response": "Short.",
            "model": llm_router.MEDIUM_MODEL,
            "latency_sec": 0.22,
            "prompt_tokens": 12,
            "completion_tokens": 1,
            "total_tokens": 13,
        }

        prompt = (
            "Analyze and compare dynamic programming algorithm complexity in detail with examples."
        )
        result = llm_router.route_request(prompt)

        self.assertFalse(result["retried"])
        self.assertEqual(mock_generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
