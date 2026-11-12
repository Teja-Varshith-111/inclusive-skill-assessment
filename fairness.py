"""
fairness.py
===========
Fairness Evaluation Module for the Inclusive Assessment and Skill-Evaluation
Decision Support System.

Analyzes algorithmic bias and performance disparities across demographic or
institutional assessment cohorts (e.g., Group A vs Group B).
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import FEATURE_COLUMNS, CATEGORIES


def evaluate_fairness_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual_skill",
    pred_col: str = "predicted_skill",
    group_col: str = "assessment_group",
    fairness_tolerance: float = 0.10
) -> Dict[str, Any]:
    """
    Evaluates group-level fairness metrics including group accuracy,
    demographic parity, and flags any disparity exceeding the tolerance threshold.

    Parameters:
        df: DataFrame containing actual and predicted labels along with assessment group.
        actual_col: Column name for actual ground-truth label.
        pred_col: Column name for model predicted label.
        group_col: Column name for demographic/institutional cohort.
        fairness_tolerance: Maximum allowable difference (e.g. 0.10 for 10%).

    Returns:
        dict: Detailed fairness statistics, group comparisons, and alert status.
    """
    groups = sorted(df[group_col].dropna().unique())
    group_stats = {}

    # High-skill tiers considered "positive outcome" for demographic parity
    high_skill_classes = {"Advanced", "Expert"}

    for g in groups:
        sub_df = df[df[group_col] == g]
        total_in_group = len(sub_df)

        if total_in_group == 0:
            continue

        acc = accuracy_score(sub_df[actual_col], sub_df[pred_col])
        prec = precision_score(sub_df[actual_col], sub_df[pred_col], average="weighted", zero_division=0)
        rec = recall_score(sub_df[actual_col], sub_df[pred_col], average="weighted", zero_division=0)
        f1 = f1_score(sub_df[actual_col], sub_df[pred_col], average="weighted", zero_division=0)

        # Proportion of students predicted as Advanced or Expert
        high_skill_rate = (sub_df[pred_col].isin(high_skill_classes)).mean()

        # Mean skill score in this group
        available_feats = [col for col in FEATURE_COLUMNS if col in sub_df.columns]
        mean_scores = sub_df[available_feats].mean().to_dict() if available_feats else {}
        overall_mean = np.mean(list(mean_scores.values())) if mean_scores else 0.0

        # Distribution of predictions
        pred_distribution = sub_df[pred_col].value_counts(normalize=True).to_dict()
        for cat in CATEGORIES:
            pred_distribution.setdefault(cat, 0.0)

        group_stats[g] = {
            "count": total_in_group,
            "percentage_of_total": total_in_group / len(df),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "high_skill_selection_rate": float(high_skill_rate),
            "overall_mean_score": float(overall_mean),
            "skill_means": mean_scores,
            "distribution": pred_distribution
        }

    # Calculate disparities between groups (if at least 2 groups present)
    disparities = {}
    is_fair = True
    warnings: List[str] = []

    if len(groups) >= 2:
        g1, g2 = groups[0], groups[1]
        acc_diff = abs(group_stats[g1]["accuracy"] - group_stats[g2]["accuracy"])
        selection_diff = abs(group_stats[g1]["high_skill_selection_rate"] - group_stats[g2]["high_skill_selection_rate"])

        disparities = {
            "group_pair": f"{g1} vs {g2}",
            "accuracy_difference": float(acc_diff),
            "selection_rate_difference": float(selection_diff)
        }

        # Check if accuracy disparity exceeds threshold
        if acc_diff > fairness_tolerance:
            is_fair = False
            warnings.append(
                f"Accuracy Disparity Warning: Group accuracy difference is {acc_diff * 100:.1f}%, "
                f"which exceeds the recommended fairness threshold of {fairness_tolerance * 100:.1f}%."
            )

        # Check if demographic selection disparity exceeds threshold
        if selection_diff > (fairness_tolerance * 1.5):
            warnings.append(
                f"Demographic Parity Notice: High-skill prediction rate difference is {selection_diff * 100:.1f}% "
                f"between {g1} ({group_stats[g1]['high_skill_selection_rate']*100:.1f}%) and "
                f"{g2} ({group_stats[g2]['high_skill_selection_rate']*100:.1f}%)."
            )

    return {
        "group_stats": group_stats,
        "disparities": disparities,
        "is_fair": is_fair,
        "warnings": warnings,
        "fairness_tolerance": fairness_tolerance,
        "total_evaluated": len(df)
    }


def get_fairness_educational_summary() -> str:
    """
    Returns beginner-friendly educational text on why fairness matters in AI.
    """
    return """
### Why Algorithmic Fairness Matters in Education

1. **Equitable Evaluation**: AI models learn patterns from training data. If historical data reflects differences in educational access, hardware, or prior exposure, the AI can inadvertently reinforce those inequalities.
2. **Preventing Disparate Impact**: Two groups of students with similar foundational potential should experience similar accuracy and evaluation quality regardless of demographic group, batch, or background.
3. **Transparency Over Black-Box Decisions**: Tracking group-level metrics ensures educational leaders can identify systemic biases early, rather than trusting unverified algorithms blindly.
4. **Human-AI Collaboration as a Safety Valve**: Algorithmic predictions are decision aids, never the final authority. Human educators can step in to review borderline cases and correct algorithmic oversight.
"""


if __name__ == "__main__":
    from model import train_model
    _, _, test_df, _ = train_model()
    results = evaluate_fairness_metrics(test_df)
    print("Fairness Evaluation Results:")
    for group, stats in results["group_stats"].items():
        print(f"Group: {group}")
        print(f"  Count: {stats['count']}")
        print(f"  Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"  High-Skill Rate: {stats['high_skill_selection_rate']*100:.2f}%")
    print("\nWarnings:")
    for w in results["warnings"]:
        print(f"  - {w}")

