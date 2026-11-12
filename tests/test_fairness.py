"""
tests/test_fairness.py
======================
Unit tests for Fairness Evaluation, demographic parity, group accuracy disparity,
and fairness threshold warning triggers.
"""

import unittest
import pandas as pd
from fairness import evaluate_fairness_metrics, get_fairness_educational_summary


class TestFairnessEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a synthetic test DataFrame with two assessment groups
        self.balanced_df = pd.DataFrame({
            "actual_skill": ["Beginner"] * 50 + ["Advanced"] * 50,
            "predicted_skill": ["Beginner"] * 46 + ["Intermediate"] * 4 + ["Advanced"] * 47 + ["Intermediate"] * 3,
            "assessment_group": ["Group A"] * 50 + ["Group B"] * 50,
            "programming": [40] * 50 + [75] * 50,
            "problem_solving": [40] * 50 + [75] * 50,
            "communication": [40] * 50 + [75] * 50,
            "logical_reasoning": [40] * 50 + [75] * 50,
            "technical_knowledge": [40] * 50 + [75] * 50,
            "practical_application": [40] * 50 + [75] * 50,
        })

    def test_01_two_group_fairness_computation(self):
        """Verify fairness metrics calculate group stats and disparities for two groups."""
        res = evaluate_fairness_metrics(self.balanced_df, fairness_tolerance=0.10)
        self.assertIn("group_stats", res)
        self.assertIn("Group A", res["group_stats"])
        self.assertIn("Group B", res["group_stats"])
        self.assertIn("accuracy_difference", res["disparities"])
        self.assertIn("selection_rate_difference", res["disparities"])

    def test_02_disparity_warning_trigger(self):
        """Verify disparity warning is triggered when accuracy difference exceeds tolerance."""
        # Intentionally create high disparity: Group A has 100% accuracy, Group B has 50% accuracy
        skewed_df = pd.DataFrame({
            "actual_skill": ["Advanced"] * 20 + ["Advanced"] * 20,
            "predicted_skill": ["Advanced"] * 20 + ["Beginner"] * 10 + ["Advanced"] * 10,
            "assessment_group": ["Group A"] * 20 + ["Group B"] * 20,
            "programming": [75] * 40,
            "problem_solving": [75] * 40,
            "communication": [75] * 40,
            "logical_reasoning": [75] * 40,
            "technical_knowledge": [75] * 40,
            "practical_application": [75] * 40,
        })
        res = evaluate_fairness_metrics(skewed_df, fairness_tolerance=0.10)
        self.assertFalse(res["is_fair"])
        self.assertTrue(len(res["warnings"]) > 0)
        self.assertTrue(any("Accuracy Disparity Warning" in w for w in res["warnings"]))

    def test_03_single_group_graceful_handling(self):
        """Verify that when only one group exists, the module does not crash."""
        single_group_df = self.balanced_df[self.balanced_df["assessment_group"] == "Group A"]
        res = evaluate_fairness_metrics(single_group_df, fairness_tolerance=0.10)
        self.assertEqual(len(res["group_stats"]), 1)
        self.assertEqual(res["disparities"], {})
        self.assertTrue(res["is_fair"])

    def test_04_educational_summary_presence(self):
        """Verify educational text is informative and non-empty."""
        edu_text = get_fairness_educational_summary()
        self.assertIsInstance(edu_text, str)
        self.assertIn("Why Algorithmic Fairness Matters in Education", edu_text)
        self.assertIn("Disparate Impact", edu_text)


if __name__ == "__main__":
    unittest.main()

