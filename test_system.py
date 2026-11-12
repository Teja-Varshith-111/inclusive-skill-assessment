
"""
test_system.py
==============
Automated Test Suite for the Inclusive Assessment and Skill-Evaluation
Decision Support System.

Verifies:
1. Dataset generation integrity and schema
2. Model training, predictions, and XAI output
3. Fairness evaluation calculations
4. SQLite database persistence and retrieval
"""

import os
import unittest
import pandas as pd
import numpy as np

from dataset import generate_student_data, FEATURE_COLUMNS, TARGET_COLUMN, CATEGORIES
from model import train_model, predict_student_skill, explain_prediction
from fairness import evaluate_fairness_metrics
from database import init_db, save_assessment, load_all_assessments, get_connection


class TestInclusiveAssessmentSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_assessments.db")
        if os.path.exists(cls.test_db):
            os.remove(cls.test_db)
        init_db(cls.test_db)

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        try:
            if os.path.exists(cls.test_db):
                os.remove(cls.test_db)
        except Exception:
            pass

    def test_01_dataset_generation(self):
        """Verify synthetic dataset produces valid records and distributions."""
        df = generate_student_data(n_samples=100, random_seed=42)
        self.assertEqual(len(df), 100)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)
            self.assertTrue((df[col] >= 0).all() and (df[col] <= 100).all())
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertTrue(set(df[TARGET_COLUMN].unique()).issubset(set(CATEGORIES)))

    def test_02_model_training_and_metrics(self):
        """Verify model achieves acceptable classification metrics (>80% accuracy)."""
        clf, metrics, test_df, cohort_df = train_model(test_size=0.2, random_state=42)
        self.assertGreater(metrics["accuracy"], 0.80)
        self.assertGreater(metrics["precision"], 0.80)
        self.assertGreater(metrics["recall"], 0.80)
        self.assertGreater(metrics["f1_score"], 0.80)
        self.assertEqual(len(metrics["feature_importances"]), len(FEATURE_COLUMNS))

    def test_03_prediction_and_xai(self):
        """Verify individual prediction and explainable AI outputs."""
        clf, metrics, test_df, cohort_df = train_model(test_size=0.2, random_state=42)
        sample_scores = {
            "programming": 88.0,
            "problem_solving": 90.0,
            "communication": 82.0,
            "logical_reasoning": 85.0,
            "technical_knowledge": 89.0,
            "practical_application": 91.0
        }
        pred = predict_student_skill(clf, sample_scores)
        self.assertIn(pred["predicted_category"], CATEGORIES)
        self.assertGreater(pred["confidence"], 0.50)
        self.assertEqual(len(pred["class_probabilities"]), len(CATEGORIES))

        exp = explain_prediction(
            sample_scores,
            pred["predicted_category"],
            metrics["feature_importances"],
            cohort_df
        )
        self.assertIn("text_summary", exp)
        self.assertTrue(len(exp["text_summary"]) > 20)
        self.assertIn("cohort_averages", exp)

    def test_04_fairness_evaluation(self):
        """Verify fairness evaluation identifies group metrics and disparities."""
        clf, metrics, test_df, cohort_df = train_model(test_size=0.2, random_state=42)
        fairness_res = evaluate_fairness_metrics(test_df)
        self.assertIn("group_stats", fairness_res)
        self.assertIn("Group A", fairness_res["group_stats"])
        self.assertIn("Group B", fairness_res["group_stats"])
        self.assertIn("accuracy_difference", fairness_res["disparities"])

    def test_05_database_crud(self):
        """Verify SQLite insert and retrieval for human audit trail."""
        row_id = save_assessment(
            student_id="STU-TEST",
            student_name="Test Student",
            programming=75.0,
            problem_solving=78.0,
            communication=70.0,
            logical_reasoning=72.0,
            technical_knowledge=74.0,
            practical_application=76.0,
            assessment_group="Group A",
            ai_prediction="Advanced",
            ai_confidence=0.88,
            final_decision="Advanced",
            review_status="AI Accepted",
            evaluator_name="Test Evaluator",
            evaluator_notes="Test justification notes",
            db_path=self.test_db
        )
        self.assertGreater(row_id, 0)

        df = load_all_assessments(db_path=self.test_db)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["student_id"], "STU-TEST")
        self.assertEqual(df.iloc[0]["review_status"], "AI Accepted")


if __name__ == "__main__":
    unittest.main()
