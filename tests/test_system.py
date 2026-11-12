"""
tests/test_system.py
====================
End-to-end integration test suite verifying the complete flow across all modules:
Data Generation -> ML Training -> AI Prediction -> XAI -> Fairness -> Human Review -> SQLite Persistence.
"""

import os
import unittest
import pandas as pd

from dataset import generate_student_data, FEATURE_COLUMNS, CATEGORIES
from model import train_model, predict_student_skill, explain_prediction
from fairness import evaluate_fairness_metrics
from database import init_db, save_assessment, load_all_assessments, get_connection


class TestSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_test.db")
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass
        init_db(cls.test_db)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db):
            try:
                os.remove(cls.test_db)
            except Exception:
                pass

    def test_complete_end_to_end_flow(self):
        """Execute and verify the full student assessment lifecycle."""
        # Step 1: Synthetic Dataset Generation
        dataset_df = generate_student_data(n_samples=150, random_seed=42)
        self.assertEqual(len(dataset_df), 150)

        # Step 2: Model Training
        clf, metrics, test_df, cohort_df = train_model(test_size=0.2, random_state=42)
        self.assertGreaterEqual(metrics["accuracy"], 0.85)

        # Step 3: New Student Assessment Input & Inference
        student_scores = {
            "programming": 82.0,
            "problem_solving": 84.0,
            "communication": 78.0,
            "logical_reasoning": 80.0,
            "technical_knowledge": 85.0,
            "practical_application": 86.0
        }
        pred_res = predict_student_skill(clf, student_scores)
        self.assertIn(pred_res["predicted_category"], ["Advanced", "Expert"])
        self.assertGreater(pred_res["confidence"], 0.5)

        # Step 4: Explainable AI
        exp_res = explain_prediction(
            student_scores,
            pred_res["predicted_category"],
            metrics["feature_importances"],
            cohort_df
        )
        self.assertIn("text_summary", exp_res)
        self.assertTrue(len(exp_res["text_summary"]) > 20)

        # Step 5: Fairness Analysis on Cohort
        fairness_res = evaluate_fairness_metrics(test_df)
        self.assertIn("group_stats", fairness_res)

        # Step 6: Human-AI Collaboration (Override and Save to Database)
        # Evaluator decides to override to 'Expert' based on practical portfolio
        row_id = save_assessment(
            student_id="STU-E2E-1",
            student_name="Zoe Anderson",
            programming=student_scores["programming"],
            problem_solving=student_scores["problem_solving"],
            communication=student_scores["communication"],
            logical_reasoning=student_scores["logical_reasoning"],
            technical_knowledge=student_scores["technical_knowledge"],
            practical_application=student_scores["practical_application"],
            assessment_group="Group A",
            ai_prediction=pred_res["predicted_category"],
            ai_confidence=pred_res["confidence"],
            final_decision="Expert",
            review_status="Human Overridden",
            evaluator_name="Lead Evaluator",
            evaluator_notes="Elevated to Expert due to outstanding open-source contributions.",
            db_path=self.test_db
        )
        self.assertGreater(row_id, 0)

        # Step 7: Assessment History Audit Log Retrieval & Verification
        saved_records = load_all_assessments(db_path=self.test_db)
        self.assertEqual(len(saved_records), 1)
        record = saved_records.iloc[0]
        self.assertEqual(record["student_id"], "STU-E2E-1")
        self.assertEqual(record["final_decision"], "Expert")
        self.assertEqual(record["review_status"], "Human Overridden")
        self.assertIn("open-source", record["evaluator_notes"])


if __name__ == "__main__":
    unittest.main()

