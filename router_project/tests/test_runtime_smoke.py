import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("GROQ_API_KEY", "test-key")


class TestRuntimeSmoke(unittest.TestCase):
    def test_project_compiles(self):
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "app"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"compileall failed\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_groq_client_import_fails_without_api_key(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        with tempfile.TemporaryDirectory() as tmp_cwd:
            env = os.environ.copy()
            env.pop("GROQ_API_KEY", None)
            env["PYTHONPATH"] = project_root

            result = subprocess.run(
                [sys.executable, "-c", "import app.services.groq_client"],
                cwd=tmp_cwd,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GROQ_API_KEY environment variable not set", result.stderr)


if __name__ == "__main__":
    unittest.main()
