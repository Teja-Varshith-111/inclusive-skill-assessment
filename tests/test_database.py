"""
tests/test_database.py
======================
Unit tests for SQLite database management, schema creation, assessment persistence,
and human review decision audit trail.
"""

import os
import unittest
import pandas as pd
from database import init_db, save_assessment, load_all_assessments, get_connection


class TestDatabaseModule(unittest.TestCase):
    def setUp(self):
        # Use an isolated temporary database for each test
        self.test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_test_db.db")
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass
        init_db(self.test_db)

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass

    def test_01_schema_initialization(self):
        """Verify the assessments table is created with all expected columns."""
        conn = get_connection(self.test_db)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(assessments);")
            columns = [row["name"] for row in cursor.fetchall()]
            expected = [
                "id", "student_id", "student_name", "programming", "problem_solving",
                "communication", "logical_reasoning", "technical_knowledge",
                "practical_application", "assessment_group", "ai_prediction",
                "ai_confidence", "final_decision", "review_status",
                "evaluator_name", "evaluator_notes", "created_at"
            ]
            for col in expected:
                self.assertIn(col, columns)
        finally:
            conn.close()

    def test_02_save_and_retrieve_assessment(self):
        """Verify saving an assessment and reading it back preserves all field values."""
        row_id = save_assessment(
            student_id="STU-5001",
            student_name="Ananya Patel",
            programming=85.0,
            problem_solving=80.0,
            communication=75.0,
            logical_reasoning=82.0,
            technical_knowledge=84.0,
            practical_application=86.0,
            assessment_group="Group A",
            ai_prediction="Advanced",
            ai_confidence=0.91,
            final_decision="Advanced",
            review_status="AI Accepted",
            evaluator_name="Dr. Vance",
            evaluator_notes="Strong performance in coursework.",
            db_path=self.test_db
        )
        self.assertGreater(row_id, 0)

        df = load_all_assessments(db_path=self.test_db)
        self.assertEqual(len(df), 1)
        record = df.iloc[0]
        self.assertEqual(record["student_id"], "STU-5001")
        self.assertEqual(record["student_name"], "Ananya Patel")
        self.assertEqual(record["ai_prediction"], "Advanced")
        self.assertEqual(record["final_decision"], "Advanced")
        self.assertEqual(record["review_status"], "AI Accepted")

    def test_03_human_override_preserves_original_ai_prediction(self):
        """Verify that when human overrides prediction, both AI prediction and human decision are stored."""
        row_id = save_assessment(
            student_id="STU-5002",
            student_name="Carlos Garcia",
            programming=68.0,
            problem_solving=70.0,
            communication=65.0,
            logical_reasoning=67.0,
            technical_knowledge=69.0,
            practical_application=72.0,
            assessment_group="Group B",
            ai_prediction="Intermediate",
            ai_confidence=0.68,
            final_decision="Advanced",
            review_status="Human Overridden",
            evaluator_name="Prof. Sharma",
            evaluator_notes="Overridden based on practical capstone project demonstration.",
            db_path=self.test_db
        )

        df = load_all_assessments(db_path=self.test_db)
        rec = df[df["id"] == row_id].iloc[0]
        self.assertEqual(rec["ai_prediction"], "Intermediate")
        self.assertEqual(rec["final_decision"], "Advanced")
        self.assertEqual(rec["review_status"], "Human Overridden")
        self.assertIn("capstone project", rec["evaluator_notes"])

    def test_04_duplicate_student_id_allowed_as_history(self):
        """Verify the same student can have multiple assessments over time (e.g. pre and post test)."""
        save_assessment(
            student_id="STU-DUP", student_name="David Miller",
            programming=40, problem_solving=40, communication=40,
            logical_reasoning=40, technical_knowledge=40, practical_application=40,
            assessment_group="Group A", ai_prediction="Beginner", ai_confidence=0.85,
            final_decision="Beginner", review_status="AI Accepted",
            evaluator_name="Prof. Davis", evaluator_notes="Baseline", db_path=self.test_db
        )
        save_assessment(
            student_id="STU-DUP", student_name="David Miller",
            programming=75, problem_solving=75, communication=75,
            logical_reasoning=75, technical_knowledge=75, practical_application=75,
            assessment_group="Group A", ai_prediction="Advanced", ai_confidence=0.88,
            final_decision="Advanced", review_status="AI Accepted",
            evaluator_name="Prof. Davis", evaluator_notes="Post-course evaluation", db_path=self.test_db
        )

        df = load_all_assessments(db_path=self.test_db)
        student_records = df[df["student_id"] == "STU-DUP"]
        self.assertEqual(len(student_records), 2)

    def test_05_empty_database_returns_empty_dataframe(self):
        """Verify that querying an initialized but empty database returns an empty DataFrame gracefully."""
        df = load_all_assessments(db_path=self.test_db)
        self.assertTrue(df.empty)
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()

