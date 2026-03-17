import re
import unittest
from pathlib import Path


class TestFrontendStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend_file = Path(__file__).resolve().parents[2] / "chatbot_frontend" / "index.html"
        cls.html = cls.frontend_file.read_text(encoding="utf-8")

    def test_frontend_file_exists(self):
        self.assertTrue(self.frontend_file.exists(), f"Missing frontend file: {self.frontend_file}")

    def test_required_elements_and_api_path_exist(self):
        self.assertIn("id=\"messages\"", self.html)
        self.assertIn("id=\"prompt\"", self.html)
        self.assertIn("id=\"send-btn\"", self.html)
        self.assertIn("http://localhost:8000/api/chat/", self.html)

    def test_no_multiple_id_attributes_on_single_tag(self):
        has_multiple_ids = re.search(
            r"<[^>]*\bid\s*=\s*\"[^\"]+\"[^>]*\bid\s*=\s*\"[^\"]+\"[^>]*>",
            self.html,
            flags=re.IGNORECASE,
        )
        self.assertIsNone(
            has_multiple_ids,
            "Found an HTML tag containing more than one id attribute, which is invalid markup.",
        )


if __name__ == "__main__":
    unittest.main()
