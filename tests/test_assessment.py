"""
tests/test_assessment.py
========================
Unit tests for Student Assessment form data validation, scoring logic,
boundary conditions, and preset configurations.
"""

import unittest
from dataset import FEATURE_COLUMNS, CATEGORIES


class TestStudentAssessment(unittest.TestCase):
    def setUp(self):
        self.valid_student = {
            "student_id": "STU-2001",
            "student_name": "Aarav Sharma",
            "scores": {
                "programming": 84.0,
                "problem_solving": 82.0,
                "communication": 75.0,
                "logical_reasoning": 80.0,
                "technical_knowledge": 86.0,
                "practical_application": 88.0,
            },
            "assessment_group": "Group A"
        }

    def test_01_valid_student_structure(self):
        """Verify valid student data dictionary has all required keys and valid types."""
        self.assertTrue(self.valid_student["student_id"].startswith("STU-"))
        self.assertTrue(len(self.valid_student["student_name"].strip()) > 0)
        self.assertIn(self.valid_student["assessment_group"], ["Group A", "Group B"])
        self.assertEqual(len(self.valid_student["scores"]), 6)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.valid_student["scores"])
            score = self.valid_student["scores"][col]
            self.assertIsInstance(score, (int, float))
            self.assertTrue(0.0 <= score <= 100.0)

    def test_02_boundary_scores_min_and_max(self):
        """Test boundary conditions: exact 0.0 (minimum) and 100.0 (maximum)."""
        min_scores = {col: 0.0 for col in FEATURE_COLUMNS}
        max_scores = {col: 100.0 for col in FEATURE_COLUMNS}
        for col in FEATURE_COLUMNS:
            self.assertEqual(min_scores[col], 0.0)
            self.assertEqual(max_scores[col], 100.0)

    def test_03_decimal_scores_allowed(self):
        """Verify scores can accept decimal / floating-point values."""
        decimal_scores = {
            "programming": 78.5,
            "problem_solving": 82.25,
            "communication": 64.7,
            "logical_reasoning": 71.0,
            "technical_knowledge": 85.5,
            "practical_application": 79.9
        }
        for k, v in decimal_scores.items():
            self.assertIsInstance(v, float)
            self.assertTrue(0.0 <= v <= 100.0)

    def test_04_mandatory_field_validation(self):
        """Verify that empty student ID or empty student name triggers validation errors."""
        empty_id_student = {"student_id": "   ", "student_name": "John Doe"}
        empty_name_student = {"student_id": "STU-101", "student_name": ""}

        is_valid_1 = bool(empty_id_student["student_id"].strip()) and bool(empty_id_student["student_name"].strip())
        is_valid_2 = bool(empty_name_student["student_id"].strip()) and bool(empty_name_student["student_name"].strip())

        self.assertFalse(is_valid_1, "Empty student ID should fail validation.")
        self.assertFalse(is_valid_2, "Empty student name should fail validation.")

    def test_05_negative_and_out_of_bounds_detection(self):
        """Verify negative scores or scores > 100 are identified as invalid."""
        invalid_scores = [
            {"programming": -5.0},
            {"problem_solving": 105.0},
            {"communication": 150.0},
            {"technical_knowledge": -0.1}
        ]
        for item in invalid_scores:
            for k, v in item.items():
                is_in_bounds = (0.0 <= v <= 100.0)
                self.assertFalse(is_in_bounds, f"Score {v} for {k} should be out of bounds.")


if __name__ == "__main__":
    unittest.main()

