"""
dataset.py
==========
Synthetic Dataset Generator for the Inclusive Assessment and Skill-Evaluation
Decision Support System.

Generates realistic student assessment data with 6 core evaluation metrics,
an assessment group tag (for fairness evaluation), and derived skill level labels.
"""

import os
import numpy as np
import pandas as pd

# List of first and last names for generating synthetic student records
FIRST_NAMES = [
    "Aarav", "Ananya", "Alex", "Brenda", "Carlos", "Chloe", "David", "Deepa",
    "Elena", "Farhan", "Grace", "Hannah", "Ishaan", "Jasmine", "Kevin", "Lakshmi",
    "Michael", "Neha", "Oliver", "Priya", "Rahul", "Sara", "Tariq", "Uma",
    "Victor", "Wendy", "Yusuf", "Zoe", "Aditya", "Bhavna", "Dev", "Fatima"
]

LAST_NAMES = [
    "Sharma", "Patel", "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis"
]

FEATURE_COLUMNS = [
    "programming",
    "problem_solving",
    "communication",
    "logical_reasoning",
    "technical_knowledge",
    "practical_application",
]

TARGET_COLUMN = "skill_level"
CATEGORIES = ["Beginner", "Intermediate", "Advanced", "Expert"]


def generate_student_data(n_samples: int = 800, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic student assessment records.

    Parameters:
        n_samples (int): Number of student records to generate.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing student assessment details and skill levels.
    """
    np.random.seed(random_seed)

    records = []

    for i in range(1, n_samples + 1):
        student_id = f"STU-{1000 + i}"
        first = np.random.choice(FIRST_NAMES)
        last = np.random.choice(LAST_NAMES)
        full_name = f"{first} {last}"

        # Assign an assessment group (e.g. Group A / Group B)
        # We assign ~55% to Group A and ~45% to Group B
        group = "Group A" if np.random.rand() < 0.55 else "Group B"

        # Baseline capability level: 4 tiers roughly distributed across student population
        tier = np.random.choice(["low", "medium", "high", "top"], p=[0.20, 0.40, 0.25, 0.15])

        if tier == "low":
            base_mean = 35
        elif tier == "medium":
            base_mean = 55
        elif tier == "high":
            base_mean = 73
        else:
            base_mean = 88

        # Add variation per skill metric (clipped between 0 and 100)
        prog = np.clip(np.random.normal(base_mean, 10), 5, 100)
        prob = np.clip(np.random.normal(base_mean, 10), 5, 100)
        comm = np.clip(np.random.normal(base_mean, 12), 5, 100)
        logic = np.clip(np.random.normal(base_mean, 10), 5, 100)
        tech = np.clip(np.random.normal(base_mean, 11), 5, 100)
        prac = np.clip(np.random.normal(base_mean, 10), 5, 100)

        # Composite weighted score calculation:
        # Technical & Problem Solving have slightly higher weight
        composite = (
            0.22 * prog +
            0.20 * prob +
            0.16 * logic +
            0.15 * tech +
            0.15 * prac +
            0.12 * comm
        )

        # Realistic noise representing assessment day conditions (+/- a few points)
        noise = np.random.normal(0, 2.5)
        final_score = np.clip(composite + noise, 0, 100)

        # Determine ground-truth skill level category
        if final_score < 45:
            skill = "Beginner"
        elif final_score < 65:
            skill = "Intermediate"
        elif final_score < 82:
            skill = "Advanced"
        else:
            skill = "Expert"

        records.append({
            "student_id": student_id,
            "student_name": full_name,
            "programming": round(float(prog), 1),
            "problem_solving": round(float(prob), 1),
            "communication": round(float(comm), 1),
            "logical_reasoning": round(float(logic), 1),
            "technical_knowledge": round(float(tech), 1),
            "practical_application": round(float(prac), 1),
            "assessment_group": group,
            "skill_level": skill
        })

    df = pd.DataFrame(records)
    return df


def generate_and_save_dataset(filepath: str = "students_dataset.csv", n_samples: int = 800) -> pd.DataFrame:
    """Generates the synthetic dataset and saves it to a CSV file."""
    df = generate_student_data(n_samples=n_samples)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created at '{filepath}' with {len(df)} records.")
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "students_dataset.csv")
    df = generate_and_save_dataset(output_path, n_samples=800)
    print("\nSample records:")
    print(df.head(5))
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())

