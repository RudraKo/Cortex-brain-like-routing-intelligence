import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.services import prompt_practice


class TestPromptPracticeUnit(unittest.TestCase):
    def test_get_full_course_curriculum_contains_all_levels(self):
        course = prompt_practice.get_full_course_curriculum()
        self.assertEqual(course["id"], "prompt-engineering-mastery")
        self.assertGreaterEqual(len(course["levels"]), 5)
        self.assertEqual(course["levels"][0]["level"], "Beginner")
        self.assertEqual(course["levels"][-1]["level"], "Industry Expert")

    def test_analyze_prompt_quality_returns_expected_shape(self):
        result = prompt_practice.analyze_prompt_quality(
            "Act as a product strategist. Compare pricing models for B2B SaaS and output a table."
        )

        self.assertIn("scores", result)
        self.assertIn("overall_score", result)
        self.assertIn("rewritten_prompt", result)
        self.assertIn("detected_task_type", result)
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 10.0)
        self.assertIn("Task:", result["rewritten_prompt"])

    @patch("app.services.prompt_practice.route_request")
    def test_run_prompt_test_ranks_higher_scored_prompt(self, mock_route):
        mock_route.side_effect = [
            {
                "response": "Short answer.",
                "model": "llama-3.1-8b-instant",
                "latency_sec": 0.10,
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "complexity_score": 0.1,
                "task_type": "factual",
                "retried": False,
                "from_cache": False,
            },
            {
                "response": "1. Launch message\n2. Audience targeting\n3. KPI plan\n4. Risks",
                "model": "llama-3.1-70b-versatile",
                "latency_sec": 0.32,
                "prompt_tokens": 12,
                "completion_tokens": 24,
                "total_tokens": 36,
                "complexity_score": 0.4,
                "task_type": "creative",
                "retried": False,
                "from_cache": False,
            },
        ]

        result = prompt_practice.run_prompt_test(
            scenario="Create an AI product launch plan for students.",
            prompts=[
                "Give a short launch plan.",
                "Provide step-by-step launch plan with metrics and bullets.",
            ],
        )

        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(result["winner_index"], 1)
        self.assertEqual(result["ranking"][0], 1)


if __name__ == "__main__":
    unittest.main()
