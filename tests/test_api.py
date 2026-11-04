"""
Integration tests for the ContextIQ FastAPI REST API.

Tests cover all endpoints: health, ask (answered + refused),
ingest, reload, evaluate scorecard, and vector search.
"""
import unittest
from fastapi.testclient import TestClient

from api import app


class TestContextIQAPI(unittest.TestCase):
    """Integration tests for ContextIQ REST endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def test_health_returns_200(self):
        """GET /health returns 200 with healthy status and stats."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("stats", data)
        self.assertGreater(data["stats"]["document_count"], 0)
        self.assertGreater(data["stats"]["chunk_count"], 0)

    # ------------------------------------------------------------------
    # Ask — Answered
    # ------------------------------------------------------------------
    def test_ask_password_reset_returns_answered(self):
        """POST /api/v1/ask with password reset question returns ANSWERED + D-01."""
        resp = self.client.post("/api/v1/ask", json={"question": "How do I reset my password?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ANSWERED")
        self.assertIsNotNone(data.get("citation"))
        self.assertEqual(data["citation"]["doc_id"], "D-01")

    def test_ask_csv_limit_returns_answered(self):
        """POST /api/v1/ask with CSV limit question returns ANSWERED + D-03."""
        resp = self.client.post("/api/v1/ask", json={"question": "What's the maximum number of rows I can export to CSV?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ANSWERED")
        self.assertEqual(data["citation"]["doc_id"], "D-03")

    # ------------------------------------------------------------------
    # Ask — Refused
    # ------------------------------------------------------------------
    def test_ask_salesforce_returns_refused(self):
        """POST /api/v1/ask with Salesforce question returns REFUSED."""
        resp = self.client.post("/api/v1/ask", json={"question": "Can I connect to Salesforce?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "REFUSED")

    def test_ask_empty_question_returns_400(self):
        """POST /api/v1/ask with empty question returns 400."""
        resp = self.client.post("/api/v1/ask", json={"question": "   "})
        self.assertEqual(resp.status_code, 400)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def test_evaluate_returns_full_accuracy(self):
        """GET /api/v1/evaluate returns 100.0% accuracy and 0.0% hallucination."""
        resp = self.client.get("/api/v1/evaluate")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["accuracy"], "100.0%")
        self.assertEqual(data["hallucination_rate"], "0.0%")
        self.assertEqual(data["passed"], data["total_questions"])

    # ------------------------------------------------------------------
    # Vector Search
    # ------------------------------------------------------------------
    def test_vector_search_returns_results(self):
        """GET /api/v1/vector-search returns ranked results."""
        resp = self.client.get("/api/v1/vector-search", params={"query": "password reset", "top_k": 3})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        # Results should have scores
        for r in data["results"]:
            self.assertIn("score", r)
            self.assertIn("doc_id", r)

    def test_vector_search_with_doc_filter(self):
        """GET /api/v1/vector-search with doc_id filter returns only matching docs."""
        resp = self.client.get("/api/v1/vector-search", params={"query": "password", "top_k": 5, "doc_id": "D-01"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for r in data["results"]:
            self.assertEqual(r["doc_id"], "D-01")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def test_ingest_new_document(self):
        """POST /api/v1/ingest adds a document and returns updated chunk count."""
        resp = self.client.post("/api/v1/ingest", json={
            "doc_id": "D-TEST-API",
            "title": "API Test Document",
            "content": "This is a test document for API integration testing. It has some content.",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ingested")
        self.assertEqual(data["doc_id"], "D-TEST-API")
        self.assertGreater(data["total_chunks"], 0)

        # Cleanup: reload corpus to remove test document
        self.client.post("/api/v1/reload", json={})

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------
    def test_reload_corpus(self):
        """POST /api/v1/reload re-indexes the corpus from disk."""
        resp = self.client.post("/api/v1/reload", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "reloaded")
        self.assertGreater(data["chunk_count"], 0)


if __name__ == "__main__":
    unittest.main()
