"""
tests/test_model.py
===================
Unit tests for Machine Learning Supervised Classification, metrics calculation,
prediction functionality, and feature exclusivity.
"""

import unittest
import numpy as np
import pandas as pd
from dataset import generate_student_data, FEATURE_COLUMNS, TARGET_COLUMN, CATEGORIES
from model import train_model, predict_student_skill


class TestMachineLearningModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Train model once for testing efficiency."""
        cls.clf, cls.metrics, cls.test_df, cls.cohort_df = train_model(test_size=0.2, random_state=42)

    def test_01_dataset_generation_integrity(self):
        """Verify synthetic dataset properties, column count, and valid value ranges."""
        df = generate_student_data(n_samples=200, random_seed=42)
        self.assertEqual(len(df), 200)
        expected_cols = [
            "student_id", "student_name", "programming", "problem_solving",
            "communication", "logical_reasoning", "technical_knowledge",
            "practical_application", "assessment_group", "skill_level"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check skill level values are valid categories
        unique_cats = set(df["skill_level"].unique())
        self.assertTrue(unique_cats.issubset(set(CATEGORIES)))

    def test_02_model_performance_benchmarks(self):
        """Verify model achieves college-grade benchmark accuracy (>85%)."""
        self.assertGreaterEqual(self.metrics["accuracy"], 0.85)
        self.assertGreaterEqual(self.metrics["precision"], 0.85)
        self.assertGreaterEqual(self.metrics["recall"], 0.85)
        self.assertGreaterEqual(self.metrics["f1_score"], 0.85)

    def test_03_features_exclude_sensitive_metadata(self):
        """Strict ethical AI test: Verify model trained ONLY on 6 skills, NOT name, ID, or group."""
        self.assertEqual(self.clf.n_features_in_, 6)
        self.assertEqual(list(self.clf.feature_names_in_), FEATURE_COLUMNS)
        self.assertNotIn("student_id", self.clf.feature_names_in_)
        self.assertNotIn("student_name", self.clf.feature_names_in_)
        self.assertNotIn("assessment_group", self.clf.feature_names_in_)

    def test_04_prediction_all_four_tiers(self):
        """Verify model correctly predicts across all 4 skill tiers with realistic score vectors."""
        test_cases = [
            ({"programming": 25, "problem_solving": 20, "communication": 30, "logical_reasoning": 25, "technical_knowledge": 22, "practical_application": 24}, "Beginner"),
            ({"programming": 55, "problem_solving": 58, "communication": 52, "logical_reasoning": 54, "technical_knowledge": 57, "practical_application": 56}, "Intermediate"),
            ({"programming": 76, "problem_solving": 78, "communication": 72, "logical_reasoning": 75, "technical_knowledge": 77, "practical_application": 74}, "Advanced"),
            ({"programming": 95, "problem_solving": 92, "communication": 88, "logical_reasoning": 94, "technical_knowledge": 96, "practical_application": 95}, "Expert")
        ]

        for scores, expected_tier in test_cases:
            pred = predict_student_skill(self.clf, scores)
            self.assertEqual(pred["predicted_category"], expected_tier,
                             f"Scores {scores} expected {expected_tier} but got {pred['predicted_category']}")
            self.assertGreaterEqual(pred["confidence"], 0.40)

    def test_05_class_probabilities_sum_to_one(self):
        """Verify output probabilities form a valid probability distribution (sum = 1.0)."""
        sample_scores = {"programming": 60, "problem_solving": 65, "communication": 60, "logical_reasoning": 62, "technical_knowledge": 64, "practical_application": 61}
        pred = predict_student_skill(self.clf, sample_scores)
        total_prob = sum(pred["class_probabilities"].values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

