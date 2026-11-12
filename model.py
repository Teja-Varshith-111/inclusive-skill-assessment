"""
model.py
========
Machine Learning Engine for the Inclusive Assessment and Skill-Evaluation
Decision Support System.

Implements:
1. Supervised Classification (Random Forest / Decision Tree)
2. Evaluation Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
3. Explainable AI (Feature Importance & Decision Drivers)
"""

import os
import pickle
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CATEGORIES,
    generate_and_save_dataset
)

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_classifier.pkl")
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students_dataset.csv")


def load_dataset() -> pd.DataFrame:
    """Loads the student dataset from CSV or generates it if not present."""
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset not found at '{DATASET_FILE}'. Generating new dataset...")
        df = generate_and_save_dataset(DATASET_FILE, n_samples=800)
    else:
        df = pd.read_csv(DATASET_FILE)
    return df


def train_model(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Trains a Random Forest classifier on the student assessment dataset.

    Returns:
        tuple: (trained_model, metrics_dict, test_features_with_group, test_labels)
    """
    df = load_dataset()

    # Features: STRICTLY assessment skill scores only (No names, IDs, or demographic groups)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Keep track of assessment group in test split for fairness evaluation
    groups = df["assessment_group"]

    # Stratified split to ensure balanced representation across categories
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train Random Forest Classifier
    # Max depth is kept moderate (6) to prevent overfitting and ensure interpretability
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    # Predictions for evaluation
    y_pred = clf.predict(X_test)

    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
    report = classification_report(y_test, y_pred, labels=CATEGORIES, output_dict=True)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_importances": dict(zip(FEATURE_COLUMNS, clf.feature_importances_))
    }

    # Save trained model to disk
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": clf, "metrics": metrics}, f)

    # Create a test dataframe that includes assessment_group for fairness evaluation
    test_df = X_test.copy()
    test_df["actual_skill"] = y_test
    test_df["predicted_skill"] = y_pred
    test_df["assessment_group"] = g_test

    return clf, metrics, test_df, df


def load_or_train_model() -> Tuple[RandomForestClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Loads an existing trained model and dataset, or trains a new one if not saved.
    """
    df = load_dataset()
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                saved = pickle.load(f)
                clf = saved["model"]
                metrics = saved["metrics"]
                return clf, metrics, df
        except Exception:
            pass

    # If not found or failed, retrain
    clf, metrics, _, df = train_model()
    return clf, metrics, df


def predict_student_skill(
    model: RandomForestClassifier,
    scores: Dict[str, float]
) -> Dict[str, Any]:
    """
    Takes a dictionary of 6 skill scores and returns the predicted skill level,
    confidence probability, and full class probability distribution.
    """
    input_df = pd.DataFrame([scores])[FEATURE_COLUMNS]
    predicted_category = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Map probability to classes
    class_probs = {cls_name: float(prob) for cls_name, prob in zip(model.classes_, probabilities)}
    confidence = float(max(probabilities))

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "class_probabilities": class_probs
    }


def explain_prediction(
    scores: Dict[str, float],
    predicted_category: str,
    feature_importances: Dict[str, float],
    cohort_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Explainable AI (XAI) engine:
    Breaks down why the AI predicted a specific skill level by comparing
    the student's scores with feature importances and cohort averages.

    Returns:
        dict: Explanations, top positive drivers, areas for growth, and comparisons.
    """
    avg_scores = cohort_df[FEATURE_COLUMNS].mean().to_dict()

    score_differences = {}
    for feat in FEATURE_COLUMNS:
        score_differences[feat] = scores[feat] - avg_scores[feat]

    # Rank features by a combination of model importance and how strongly the student performed
    ranked_factors = sorted(
        FEATURE_COLUMNS,
        key=lambda f: abs(score_differences[f]) * feature_importances.get(f, 0.1),
        reverse=True
    )

    top_strengths = [
        f for f in ranked_factors if scores[f] >= 65 and score_differences[f] >= 0
    ][:3]

    growth_areas = [
        f for f in ranked_factors if scores[f] < 60 or score_differences[f] < 0
    ][:3]

    # Human-readable textual summary
    composite_avg = sum(scores.values()) / len(scores)

    if predicted_category == "Expert":
        text_explanation = (
            f"The student demonstrated stellar mastery across technical and problem-solving dimensions "
            f"(Average score: {composite_avg:.1f}/100). The most decisive factors contributing to the "
            f"Expert classification are exceptionally high scores in "
            f"{', '.join([f.replace('_', ' ').title() for f in top_strengths] or ['core technical competencies'])}."
        )
    elif predicted_category == "Advanced":
        text_explanation = (
            f"The student exhibits solid proficiency with consistent performance "
            f"(Average score: {composite_avg:.1f}/100). High performance in key pillars such as "
            f"{', '.join([f.replace('_', ' ').title() for f in top_strengths] or ['practical concepts'])} "
            f"placed the student comfortably in the Advanced bracket."
        )
    elif predicted_category == "Intermediate":
        text_explanation = (
            f"The student shows a competent foundational understanding (Average score: {composite_avg:.1f}/100). "
            f"While core abilities are established, further development in "
            f"{', '.join([f.replace('_', ' ').title() for f in growth_areas] or ['specialized problem solving'])} "
            f"will elevate their capability to the Advanced tier."
        )
    else:  # Beginner
        text_explanation = (
            f"The student is currently building essential foundational knowledge (Average score: {composite_avg:.1f}/100). "
            f"The classification reflects that foundational scores in "
            f"{', '.join([f.replace('_', ' ').title() for f in growth_areas] or ['programming and logic'])} "
            f"require guided reinforcement and structured practice."
        )

    return {
        "text_summary": text_explanation,
        "composite_avg": composite_avg,
        "cohort_averages": avg_scores,
        "score_differences": score_differences,
        "top_strengths": top_strengths,
        "growth_areas": growth_areas
    }


if __name__ == "__main__":
    clf, metrics, test_df, df = train_model()
    print("Model Training Completed.")
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Weighted Recall:    {metrics['recall'] * 100:.2f}%")
    print(f"Weighted F1-Score:  {metrics['f1_score'] * 100:.2f}%")
    print("\nFeature Importances:")
    for k, v in sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k:22s}: {v*100:5.2f}%")

