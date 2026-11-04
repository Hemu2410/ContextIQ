"""
Automated Unit and Benchmark Regression Test Suite for ContextIQ.
"""
import os
import unittest

from contextiq.engine import ContextIQEngine
from contextiq.models import Document


class TestContextIQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs_path = os.path.join("starter_kit", "data", "docs.json")
        cls.engine = ContextIQEngine(cls.docs_path)

    def test_q1_password_reset(self):
        resp = self.engine.ask("How do I reset my password?")
        self.assertEqual(resp.status, "ANSWERED")
        self.assertIsNotNone(resp.citation)
        self.assertEqual(resp.citation.doc_id, "D-01")
        self.assertEqual(resp.citation.title, "Resetting Your Password")
        self.assertIn("password", resp.answer.lower())

    def test_q2_password_expiry(self):
        resp = self.engine.ask("How long do I have to use a password reset link before it expires?")
        self.assertEqual(resp.status, "ANSWERED")
        self.assertIsNotNone(resp.citation)
        self.assertEqual(resp.citation.doc_id, "D-01")
        self.assertIn("15 minutes", resp.answer.lower())

    def test_q3_csv_export_limit(self):
        resp = self.engine.ask("What's the maximum number of rows I can export to CSV?")
        self.assertEqual(resp.status, "ANSWERED")
        self.assertIsNotNone(resp.citation)
        self.assertEqual(resp.citation.doc_id, "D-03")
        self.assertIn("10,000", resp.answer)

    def test_q4_growth_plan_rate_limit(self):
        resp = self.engine.ask("What are the API rate limits for the Growth plan?")
        self.assertEqual(resp.status, "ANSWERED")
        self.assertIsNotNone(resp.citation)
        self.assertEqual(resp.citation.doc_id, "D-06")
        self.assertIn("60 requests per minute", resp.answer.lower())

    def test_q5_salesforce_rejection(self):
        # Solstice docs have no Salesforce content.
        # Strict requirement: MUST REFUSE, never guess or hallucinate.
        resp = self.engine.ask("Can I connect Solstice to Salesforce?")
        self.assertEqual(resp.status, "REFUSED")
        self.assertIsNone(resp.citation)
        self.assertIn("don't have an answer", resp.answer.lower())
        self.assertEqual(resp.confidence_tier, "REFUSAL")

    def test_empty_query_rejection(self):
        resp = self.engine.ask("   ")
        self.assertEqual(resp.status, "REFUSED")
        self.assertEqual(resp.confidence_tier, "REFUSAL")

    def test_unrelated_out_of_domain_query(self):
        resp = self.engine.ask("What is the recipe for chocolate cake?")
        self.assertEqual(resp.status, "REFUSED")
        self.assertIsNone(resp.citation)

    def test_on_demand_ingestion_lifecycle(self):
        # 1. Query about Zapier fails initially
        resp_before = self.engine.ask("Can I connect Solstice to Zapier?")
        self.assertEqual(resp_before.status, "REFUSED")

        # 2. Ingest a new doc D-07 Zapier Integration on demand
        new_doc = Document(
            doc_id="D-07",
            title="Connecting Zapier to Solstice",
            content="Admins can connect Solstice to Zapier from Settings > Integrations > Zapier. Enter your API key to trigger automated webhooks.",
        )
        self.engine.ingest_document(new_doc)

        # 3. Query about Zapier now succeeds with verified citation!
        resp_after = self.engine.ask("Can I connect Solstice to Zapier?")
        self.assertEqual(resp_after.status, "ANSWERED")
        self.assertIsNotNone(resp_after.citation)
        self.assertEqual(resp_after.citation.doc_id, "D-07")
        self.assertIn("Zapier", resp_after.answer)

        # 4. Clean up / restore original state for test isolation
        self.engine.load_docs(self.docs_path)


    def test_first_word_oov_entity(self):
        # Even when the external entity is at position 0, it should be rejected
        resp = self.engine.ask("Salesforce integration with Solstice is available?")
        self.assertEqual(resp.status, "REFUSED")
        self.assertIsNone(resp.citation)

    def test_qdrant_vector_search(self):
        # Verify Qdrant vector store indexing and retrieval
        results = self.engine.vector_search("password reset link", top_k=1)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].chunk.doc_id, "D-01")
        self.assertIn("password", results[0].payload["title"].lower())

    def test_intent_classification(self):
        evaluator = self.engine.evaluator
        q_quant = evaluator.classify_intent("How long do I have before link expires?")
        self.assertEqual(q_quant.intent.value, "QUANTITATIVE")

        q_proc = evaluator.classify_intent("How do I reset my password?")
        self.assertEqual(q_proc.intent.value, "PROCEDURAL")

        q_info = evaluator.classify_intent("What is the billing cycle?")
        self.assertEqual(q_info.intent.value, "INFORMATIONAL")


    def test_disambiguation_near_tie(self):
        # Ambiguous query without distinguishing noun triggers DISAMBIGUATION
        resp = self.engine.ask("How do I reset?")
        self.assertEqual(resp.status, "DISAMBIGUATION")
        self.assertIsNotNone(resp.session_id)
        self.assertIsNotNone(resp.options)
        self.assertEqual(len(resp.options), 2)
        self.assertEqual(resp.options[0]["option"], "A")
        self.assertEqual(resp.options[0]["doc_id"], "D-01")
        self.assertEqual(resp.options[1]["option"], "B")
        self.assertEqual(resp.options[1]["doc_id"], "D-04")

    def test_disambiguation_follow_up_resolution(self):
        # Turn 1: Ask ambiguous query and receive session_id
        resp1 = self.engine.ask("How do I reset?")
        self.assertEqual(resp1.status, "DISAMBIGUATION")
        sid = resp1.session_id

        # Turn 2: Follow up with "A" using the same session_id
        resp2 = self.engine.ask("A", session_id=sid)
        self.assertEqual(resp2.status, "ANSWERED")
        self.assertIsNotNone(resp2.citation)
        self.assertEqual(resp2.citation.doc_id, "D-01")
        self.assertIn("password", resp2.answer.lower())

    def test_unambiguous_query_skips_disambiguation(self):
        # Specific query must directly answer without triggering disambiguation
        resp = self.engine.ask("How do I reset my password?")
        self.assertEqual(resp.status, "ANSWERED")
        self.assertIsNotNone(resp.citation)
        self.assertEqual(resp.citation.doc_id, "D-01")


if __name__ == "__main__":
    unittest.main()
