import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.main import app


class TestPromptPracticeApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_modules_endpoint_returns_list(self):
        response = self.client.get("/api/chat/practice/modules")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        self.assertGreater(len(response.json()), 0)

    def test_course_endpoint_returns_levels(self):
        response = self.client.get("/api/chat/practice/course")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["id"], "prompt-engineering-mastery")
        self.assertIn("levels", payload)
        self.assertGreaterEqual(len(payload["levels"]), 5)

    @patch("app.api.endpoints.chat.WRITE_API_KEY", "secret-token")
    def test_write_auth_blocks_missing_api_key_when_configured(self):
        response = self.client.post(
            "/api/chat/practice/analyze",
            json={"prompt": "Summarize this in bullets."},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("API key", response.json()["detail"])

    @patch("app.api.endpoints.chat.analyze_prompt_quality")
    def test_analyze_endpoint_returns_payload(self, mock_analyze):
        mock_analyze.return_value = {
            "scores": {
                "clarity": 8.0,
                "context": 7.0,
                "constraints": 7.5,
                "evaluation": 6.5,
                "structure": 8.5,
            },
            "overall_score": 7.5,
            "detected_task_type": "factual",
            "detected_complexity": 0.21,
            "strengths": ["Has clear objective."],
            "improvements": ["Add explicit success criteria."],
            "rewritten_prompt": "You are a domain expert...",
            "checklist": ["State role."],
        }

        response = self.client.post(
            "/api/chat/practice/analyze",
            json={"prompt": "Summarize this article in 5 bullets."},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["overall_score"], 7.5)

    @patch("app.api.endpoints.chat.run_prompt_test")
    def test_test_endpoint_returns_winner(self, mock_test):
        mock_test.return_value = {
            "scenario": "Create launch copy",
            "winner_index": 0,
            "ranking": [0, 1],
            "runs": [
                {
                    "index": 0,
                    "prompt": "Prompt A",
                    "response": "Response A",
                    "score": 8.2,
                    "metrics": {
                        "relevance": 8.0,
                        "instruction_adherence": 8.5,
                        "structure": 8.0,
                        "conciseness": 8.0,
                    },
                    "router_meta": {
                        "model": "llama-3.1-70b-versatile",
                        "task_type": "creative",
                        "complexity_score": 0.35,
                        "latency_sec": 0.32,
                        "total_tokens": 40,
                        "from_cache": False,
                    },
                },
                {
                    "index": 1,
                    "prompt": "Prompt B",
                    "response": "Response B",
                    "score": 7.1,
                    "metrics": {
                        "relevance": 7.0,
                        "instruction_adherence": 7.2,
                        "structure": 7.0,
                        "conciseness": 7.0,
                    },
                    "router_meta": {
                        "model": "llama-3.1-8b-instant",
                        "task_type": "creative",
                        "complexity_score": 0.15,
                        "latency_sec": 0.10,
                        "total_tokens": 20,
                        "from_cache": False,
                    },
                },
            ],
        }

        response = self.client.post(
            "/api/chat/practice/test",
            json={"scenario": "Create launch copy", "prompts": ["Prompt A", "Prompt B"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["winner_index"], 0)


if __name__ == "__main__":
    unittest.main()
