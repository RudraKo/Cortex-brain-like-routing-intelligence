import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("GROQ_API_KEY", "test-key")

from app.db.database import get_db
from app.db.models import Base, LearnerProgress, RequestLog
from app.main import app


class TestChatApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(cls._tmpdir.name, "test_router.db")
        test_db_url = f"sqlite:///{db_path}"
        cls.engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

        def override_get_db():
            db = cls.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        cls._tmpdir.cleanup()

    def setUp(self):
        with self.SessionLocal() as db:
            db.query(LearnerProgress).delete()
            db.query(RequestLog).delete()
            db.commit()

    def test_create_chat_completion_persists_non_cached_result(self):
        mocked = {
            "response": "Processed response with sufficient detail.",
            "model": "llama-3.1-70b-versatile",
            "latency_sec": 0.21,
            "prompt_tokens": 14,
            "completion_tokens": 20,
            "total_tokens": 34,
            "complexity_score": 0.41,
            "task_type": "coding",
            "retried": False,
            "from_cache": False,
        }

        with patch("app.api.endpoints.chat.route_request", return_value=mocked):
            res = self.client.post("/api/chat/", json={"prompt": "Write a small API client."})

        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["model"], mocked["model"])
        self.assertFalse(payload["from_cache"])

        with self.SessionLocal() as db:
            rows = db.query(RequestLog).all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].task_type, "coding")
            self.assertEqual(rows[0].chosen_model, mocked["model"])
            self.assertFalse(rows[0].from_cache)

    def test_create_chat_completion_skips_db_write_for_cached_result(self):
        mocked = {
            "response": "Cached response.",
            "model": "llama-3.1-8b-instant",
            "latency_sec": 0.01,
            "prompt_tokens": 8,
            "completion_tokens": 6,
            "total_tokens": 14,
            "complexity_score": 0.0,
            "task_type": "factual",
            "retried": False,
            "from_cache": True,
        }

        with patch("app.api.endpoints.chat.route_request", return_value=mocked):
            res = self.client.post("/api/chat/", json={"prompt": "What is 2+2?"})

        self.assertEqual(res.status_code, 200)
        with self.SessionLocal() as db:
            count = db.query(RequestLog).count()
            self.assertEqual(count, 0)

    def test_create_chat_completion_returns_422_for_invalid_payload(self):
        res = self.client.post("/api/chat/", json={})
        self.assertEqual(res.status_code, 422)

    def test_create_chat_completion_returns_500_when_router_raises(self):
        with patch("app.api.endpoints.chat.route_request", side_effect=Exception("router failure")):
            res = self.client.post("/api/chat/", json={"prompt": "Trigger error"})

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json()["detail"], "router failure")

    def test_stats_endpoint_aggregates_by_model_and_task(self):
        with self.SessionLocal() as db:
            db.add_all(
                [
                    RequestLog(
                        prompt="p1",
                        task_type="coding",
                        complexity_score=0.4,
                        chosen_model="llama-3.1-70b-versatile",
                        latency_sec=0.2,
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        retried=True,
                        from_cache=False,
                    ),
                    RequestLog(
                        prompt="p2",
                        task_type="coding",
                        complexity_score=0.5,
                        chosen_model="llama-3.1-70b-versatile",
                        latency_sec=0.4,
                        prompt_tokens=10,
                        completion_tokens=30,
                        total_tokens=40,
                        retried=False,
                        from_cache=False,
                    ),
                    RequestLog(
                        prompt="p3",
                        task_type="factual",
                        complexity_score=0.0,
                        chosen_model="llama-3.1-8b-instant",
                        latency_sec=0.1,
                        prompt_tokens=5,
                        completion_tokens=6,
                        total_tokens=11,
                        retried=False,
                        from_cache=False,
                    ),
                ]
            )
            db.commit()

        res = self.client.get("/api/chat/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data), 2)

        stats = {(row["model"], row["task_type"]): row for row in data}
        coding = stats[("llama-3.1-70b-versatile", "coding")]
        factual = stats[("llama-3.1-8b-instant", "factual")]

        self.assertEqual(coding["total_requests"], 2)
        self.assertEqual(coding["avg_latency_sec"], 0.3)
        self.assertEqual(coding["avg_tokens"], 35.0)
        self.assertEqual(coding["total_tokens_used"], 70)
        self.assertEqual(coding["total_retries"], 1)

        self.assertEqual(factual["total_requests"], 1)
        self.assertEqual(factual["total_tokens_used"], 11)
        self.assertEqual(factual["total_retries"], 0)

    def test_progress_endpoints_store_and_return_completion(self):
        learner_id = "learner-demo-001"

        get_res = self.client.get(f"/api/chat/progress/{learner_id}")
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["completed_count"], 0)

        put_payload = {
            "completed_lessons": ["l1m1-1", "l1m1-2", "invalid-lesson-id"],
            "active_level": "level-1",
        }
        put_res = self.client.put(f"/api/chat/progress/{learner_id}", json=put_payload)
        self.assertEqual(put_res.status_code, 200)
        body = put_res.json()
        self.assertEqual(body["learner_id"], learner_id)
        self.assertEqual(body["active_level"], "level-1")
        self.assertEqual(body["completed_lessons"], ["l1m1-1", "l1m1-2"])
        self.assertEqual(body["completed_count"], 2)
        self.assertGreater(body["completion_pct"], 0.0)

        get_again = self.client.get(f"/api/chat/progress/{learner_id}")
        self.assertEqual(get_again.status_code, 200)
        self.assertEqual(get_again.json()["completed_lessons"], ["l1m1-1", "l1m1-2"])


if __name__ == "__main__":
    unittest.main()
