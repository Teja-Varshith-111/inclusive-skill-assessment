"""
database.py
===========
SQLite Database manager for the Inclusive Assessment and Skill-Evaluation
Decision Support System.

Stores student evaluation scores, AI predictions, and human evaluator reviews
to maintain an audit trail for Human-AI Collaboration.
"""

import os
import sqlite3
import pandas as pd

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assessments.db")


def get_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Returns a SQLite connection object with row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE) -> None:
    """Initializes the database schema if it does not already exist."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                programming REAL NOT NULL,
                problem_solving REAL NOT NULL,
                communication REAL NOT NULL,
                logical_reasoning REAL NOT NULL,
                technical_knowledge REAL NOT NULL,
                practical_application REAL NOT NULL,
                assessment_group TEXT NOT NULL,
                ai_prediction TEXT NOT NULL,
                ai_confidence REAL NOT NULL,
                final_decision TEXT NOT NULL,
                review_status TEXT NOT NULL,
                evaluator_name TEXT,
                evaluator_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    finally:
        conn.close()


def save_assessment(
    student_id: str,
    student_name: str,
    programming: float,
    problem_solving: float,
    communication: float,
    logical_reasoning: float,
    technical_knowledge: float,
    practical_application: float,
    assessment_group: str,
    ai_prediction: str,
    ai_confidence: float,
    final_decision: str,
    review_status: str,
    evaluator_name: str,
    evaluator_notes: str,
    db_path: str = DB_FILE
) -> int:
    """
    Saves a student assessment record including AI prediction and human review.

    Returns:
        int: The newly inserted record's row ID.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assessments (
                student_id,
                student_name,
                programming,
                problem_solving,
                communication,
                logical_reasoning,
                technical_knowledge,
                practical_application,
                assessment_group,
                ai_prediction,
                ai_confidence,
                final_decision,
                review_status,
                evaluator_name,
                evaluator_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id.strip(),
            student_name.strip(),
            float(programming),
            float(problem_solving),
            float(communication),
            float(logical_reasoning),
            float(technical_knowledge),
            float(practical_application),
            assessment_group,
            ai_prediction,
            float(ai_confidence),
            final_decision,
            review_status,
            evaluator_name.strip() if evaluator_name else "Anonymous Evaluator",
            evaluator_notes.strip() if evaluator_notes else "No notes provided."
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def load_all_assessments(db_path: str = DB_FILE) -> pd.DataFrame:
    """
    Loads all student assessment records from SQLite into a pandas DataFrame.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM assessments ORDER BY id DESC",
            conn
        )
        return df
    finally:
        conn.close()


def seed_sample_assessments_if_empty(db_path: str = DB_FILE, seed_count: int = 15) -> None:
    """
    Populates the database with a few initial reviewed records so the dashboard
    and history tables immediately showcase realistic data on first launch.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM assessments")
        count = cursor.fetchone()[0]
    finally:
        conn.close()

    if count > 0:
        return  # Already seeded or has data

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students_dataset.csv")
    if not os.path.exists(dataset_path):
        return

    df = pd.read_csv(dataset_path).head(seed_count)
    evaluator_names = ["Prof. Davis", "Dr. Vance", "Lead Mentor Elena", "Prof. Sharma"]

    for _, row in df.iterrows():
        # Simulate AI prediction and human review
        pred = row["skill_level"]
        conf = round(float(0.82 + (hash(row["student_id"]) % 15) / 100.0), 2)

        # 85% accepted, 15% adjusted by evaluator
        is_overridden = (hash(row["student_id"]) % 7 == 0)
        if is_overridden:
            possible_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
            possible_levels.remove(pred)
            final_decision = possible_levels[0]
            status = "Human Overridden"
            notes = "Adjusted level upward based on strong practical coursework demonstration."
        else:
            final_decision = pred
            status = "AI Accepted"
            notes = "Prediction aligns well with demonstrated problem solving and technical scores."

        save_assessment(
            student_id=row["student_id"],
            student_name=row["student_name"],
            programming=row["programming"],
            problem_solving=row["problem_solving"],
            communication=row["communication"],
            logical_reasoning=row["logical_reasoning"],
            technical_knowledge=row["technical_knowledge"],
            practical_application=row["practical_application"],
            assessment_group=row["assessment_group"],
            ai_prediction=pred,
            ai_confidence=conf,
            final_decision=final_decision,
            review_status=status,
            evaluator_name=evaluator_names[hash(row["student_id"]) % len(evaluator_names)],
            evaluator_notes=notes,
            db_path=db_path
        )
    print(f"Seeded {seed_count} initial records into assessments.db.")


if __name__ == "__main__":
    init_db()
    seed_sample_assessments_if_empty(seed_count=10)
    df = load_all_assessments()
    print("Database Initialized successfully.")
    print(f"Total records in DB: {len(df)}")
    print(df[["student_id", "student_name", "ai_prediction", "final_decision", "review_status"]].head())

