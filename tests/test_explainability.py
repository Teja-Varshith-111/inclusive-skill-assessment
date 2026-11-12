"""
tests/test_explainability.py
============================
Unit tests for Explainable AI (XAI), global feature importance extraction,
local score-versus-benchmark comparisons, and natural language explanations.
"""

import unittest
from dataset import FEATURE_COLUMNS, CATEGORIES
from model import train_model, explain_prediction


class TestExplainableAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf, cls.metrics, cls.test_df, cls.cohort_df = train_model(test_size=0.2, random_state=42)

    def test_01_global_feature_importance_validity(self):
        """Verify global feature importances are generated for all 6 competencies and sum to ~1.0."""
        fi = self.metrics["feature_importances"]
        self.assertEqual(len(fi), len(FEATURE_COLUMNS))
        for col in FEATURE_COLUMNS:
            self.assertIn(col, fi)
            self.assertGreater(fi[col], 0.0)

        total_importance = sum(fi.values())
        self.assertAlmostEqual(total_importance, 1.0, places=4)

    def test_02_local_explanation_structure(self):
        """Verify explain_prediction returns all required keys and valid benchmark metrics."""
        scores = {
            "programming": 88.0,
            "problem_solving": 85.0,
            "communication": 72.0,
            "logical_reasoning": 80.0,
            "technical_knowledge": 84.0,
            "practical_application": 86.0
        }
        exp = explain_prediction(
            scores=scores,
            predicted_category="Expert",
            feature_importances=self.metrics["feature_importances"],
            cohort_df=self.cohort_df
        )

        required_keys = [
            "text_summary", "composite_avg", "cohort_averages",
            "score_differences", "top_strengths", "growth_areas"
        ]
        for key in required_keys:
            self.assertIn(key, exp)

        self.assertIsInstance(exp["text_summary"], str)
        self.assertTrue(len(exp["text_summary"]) > 30)
        self.assertEqual(len(exp["cohort_averages"]), 6)
        self.assertEqual(len(exp["score_differences"]), 6)

    def test_03_high_scores_mapped_to_strengths(self):
        """Verify high scoring features (>65) appear in top strengths."""
        scores = {
            "programming": 95.0,
            "problem_solving": 92.0,
            "communication": 50.0,
            "logical_reasoning": 50.0,
            "technical_knowledge": 50.0,
            "practical_application": 50.0
        }
        exp = explain_prediction(
            scores=scores,
            predicted_category="Advanced",
            feature_importances=self.metrics["feature_importances"],
            cohort_df=self.cohort_df
        )
        self.assertTrue("programming" in exp["top_strengths"] or "problem_solving" in exp["top_strengths"])

    def test_04_low_scores_mapped_to_growth_areas(self):
        """Verify low scoring features (<60) appear in recommended growth areas."""
        scores = {
            "programming": 30.0,
            "problem_solving": 28.0,
            "communication": 65.0,
            "logical_reasoning": 35.0,
            "technical_knowledge": 32.0,
            "practical_application": 30.0
        }
        exp = explain_prediction(
            scores=scores,
            predicted_category="Beginner",
            feature_importances=self.metrics["feature_importances"],
            cohort_df=self.cohort_df
        )
        self.assertTrue(len(exp["growth_areas"]) > 0)


if __name__ == "__main__":
    unittest.main()

