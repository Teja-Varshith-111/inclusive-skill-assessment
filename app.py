"""
app.py
======
Inclusive Assessment and Skill-Evaluation Decision Support System
A beginner-friendly, college-level AI/ML decision support application built
with Streamlit, Scikit-learn, Pandas, Matplotlib, and SQLite.

Pages:
1. Dashboard - Overview metrics, skill distribution, and performance stats.
2. Student Assessment - Input form for student evaluation scores.
3. AI Prediction - Classification outcome with confidence probabilities.
4. Explainable AI - Feature importance & decision driver breakdown.
5. Fairness Analysis - Disparity checks across demographic assessment groups.
6. Human Review - Human-in-the-loop decision acceptance/override & feedback.
7. Assessment History - SQLite audit log of finalized evaluations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local system modules
from dataset import FEATURE_COLUMNS, CATEGORIES, generate_and_save_dataset
from database import (
    init_db,
    save_assessment,
    load_all_assessments,
    seed_sample_assessments_if_empty
)
from model import (
    load_or_train_model,
    predict_student_skill,
    explain_prediction,
    train_model
)
from fairness import evaluate_fairness_metrics, get_fairness_educational_summary

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Inclusive Skill Assessment AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern cards and badges
st.markdown("""
<style>
    /* Card container */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card-sub {
        font-size: 0.82rem;
        color: #10b981;
        margin-top: 4px;
    }

    /* Badges */
    .badge-beginner {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-intermediate {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-advanced {
        background-color: #f3e8ff;
        color: #6b21a8;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-expert {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* Concept Banner */
    .concept-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        font-size: 0.92rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions & Session Setup
# ---------------------------------------------------------
@st.cache_resource
def get_system_model_and_data():
    """Caches model and dataset loading to ensure high responsiveness."""
    init_db()
    seed_sample_assessments_if_empty(seed_count=10)
    model, metrics, df = load_or_train_model()
    return model, metrics, df


model, metrics, cohort_df = get_system_model_and_data()

# Initialize session state for active student
if "current_student" not in st.session_state:
    st.session_state.current_student = {
        "student_id": "STU-2045",
        "student_name": "Aarav Sharma",
        "scores": {
            "programming": 84.0,
            "problem_solving": 82.0,
            "communication": 75.0,
            "logical_reasoning": 80.0,
            "technical_knowledge": 86.0,
            "practical_application": 88.0,
        },
        "assessment_group": "Group A",
        "prediction_result": None,
        "explanation": None
    }


def get_skill_badge_html(skill: str) -> str:
    """Returns styled HTML badge for skill categories."""
    skill_clean = skill.strip().lower()
    if "expert" in skill_clean:
        return f'<span class="badge-expert">🌟 {skill}</span>'
    elif "advanced" in skill_clean:
        return f'<span class="badge-advanced">🚀 {skill}</span>'
    elif "intermediate" in skill_clean:
        return f'<span class="badge-intermediate">⚡ {skill}</span>'
    else:
        return f'<span class="badge-beginner">🌱 {skill}</span>'


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.markdown("## 🎓 Inclusive AI Assessor")
st.sidebar.markdown(
    "<small style='color: #64748b;'>Decision Support System v1.0</small>",
    unsafe_allow_html=True
)
st.sidebar.divider()

pages = [
    "1. Dashboard",
    "2. Student Assessment",
    "3. AI Prediction",
    "4. Explainable AI",
    "5. Fairness Analysis",
    "6. Human Review",
    "7. Assessment History"
]

selected_page = st.sidebar.radio("Navigate to:", pages, index=0)

st.sidebar.divider()
st.sidebar.markdown("### 📌 Active Student")
current_stu = st.session_state.current_student
st.sidebar.write(f"**ID:** `{current_stu['student_id']}`")
st.sidebar.write(f"**Name:** {current_stu['student_name']}")
st.sidebar.write(f"**Cohort:** {current_stu['assessment_group']}")
if current_stu["prediction_result"]:
    st.sidebar.markdown(
        f"**Prediction:** {get_skill_badge_html(current_stu['prediction_result']['predicted_category'])}",
        unsafe_allow_html=True
    )
else:
    st.sidebar.caption("No prediction computed yet.")

st.sidebar.divider()
st.sidebar.caption(f"🤖 **Model:** Random Forest (100 Trees)")
st.sidebar.caption(f"🎯 **Test Accuracy:** {metrics['accuracy']*100:.1f}%")
st.sidebar.caption(f"⚖️ **Fairness Check:** Monitored")


# ---------------------------------------------------------
# PAGE 1: DASHBOARD
# ---------------------------------------------------------
if selected_page == "1. Dashboard":
    st.title("📊 System Dashboard & Overview")
    st.markdown(
        "Welcome to the **Inclusive Assessment and Skill-Evaluation Decision Support System**. "
        "This platform helps educators assess student technical proficiencies using AI while maintaining "
        "explainability, fairness across cohorts, and human evaluator oversight."
    )

    db_records = load_all_assessments()
    total_db_records = len(db_records)
    total_synthetic = len(cohort_df)

    # Top Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Evaluations Finalized</div>
            <div class="metric-card-value">{total_db_records}</div>
            <div class="metric-card-sub">Recorded in SQLite</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Model Accuracy</div>
            <div class="metric-card-value">{metrics['accuracy']*100:.1f}%</div>
            <div class="metric-card-sub">F1-Score: {metrics['f1_score']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_score_val = cohort_df[FEATURE_COLUMNS].mean().mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Average Skill Score</div>
            <div class="metric-card-value">{avg_score_val:.1f}</div>
            <div class="metric-card-sub">Across 6 Competencies</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        accepted_ratio = 100.0
        if total_db_records > 0:
            accepted_count = (db_records["review_status"] == "AI Accepted").sum()
            accepted_ratio = (accepted_count / total_db_records) * 100.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Human Agreement</div>
            <div class="metric-card-value">{accepted_ratio:.1f}%</div>
            <div class="metric-card-sub">AI Predictions Accepted</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Distribution of Student Skill Levels")
        cat_counts = cohort_df["skill_level"].value_counts().reindex(CATEGORIES, fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]
        bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, width=0.55, edgecolor="#ffffff")
        ax.set_ylabel("Number of Students", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    with chart_col2:
        st.subheader("Average Scores by Competency")
        mean_skills = cohort_df[FEATURE_COLUMNS].mean()
        clean_labels = [c.replace("_", " ").title() for c in FEATURE_COLUMNS]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars2 = ax2.barh(clean_labels, mean_skills.values, color="#4f46e5", height=0.55)
        ax2.set_xlabel("Mean Score (0 - 100)", fontsize=10)
        ax2.set_xlim(0, 100)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="x", linestyle="--", alpha=0.3)

        for bar in bars2:
            width = bar.get_width()
            ax2.annotate(f"{width:.1f}",
                         xy=(width, bar.get_y() + bar.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=9, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("### 🧩 The 4 Core AI Pillars in this System")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1. Supervised Classification**\n\nTrains a Random Forest classifier using 6 objective skill metrics to predict skill categories without student bias.")
    with p2:
        st.markdown("**2. Explainable AI**\n\nTransparently details global feature importance and reasons behind individual predictions.")
    with p3:
        st.markdown("**3. Fairness Evaluation**\n\nMonitors accuracy and high-tier representation across assessment groups to prevent disparities.")
    with p4:
        st.markdown("**4. Human-AI Collaboration**\n\nEmpowers human educators to confirm or override AI predictions with an audited decision trail.")


# ---------------------------------------------------------
# PAGE 2: STUDENT ASSESSMENT INPUT FORM
# ---------------------------------------------------------
elif selected_page == "2. Student Assessment":
    st.title("📝 Student Assessment Evaluation Form")
    st.markdown(
        "Enter student assessment scores across the six evaluation dimensions. "
        "Once entered, click **Run AI Evaluation** to compute predictions and feature explanations."
    )

    st.markdown("""
    <div class="concept-box">
        💡 <b>AI Principle Note:</b> Student Name and Student ID are metadata fields for administrative reference. 
        In strict adherence to ethical AI design, <b>neither name nor ID is ever fed to the machine learning model</b>.
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Load for easy demonstration
    st.markdown("#### ⚡ Quick Preset Demonstrations (Optional)")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    def load_preset(name, sid, prog, prob, comm, logic, tech, prac, grp):
        st.session_state.current_student["student_name"] = name
        st.session_state.current_student["student_id"] = sid
        st.session_state.current_student["scores"] = {
            "programming": float(prog),
            "problem_solving": float(prob),
            "communication": float(comm),
            "logical_reasoning": float(logic),
            "technical_knowledge": float(tech),
            "practical_application": float(prac)
        }
        st.session_state.current_student["assessment_group"] = grp
        st.session_state.current_student["prediction_result"] = None
        st.session_state.current_student["explanation"] = None

    if preset_col1.button("🌱 Load Beginner Example"):
        load_preset("Samir Roy", "STU-3001", 32, 28, 45, 30, 35, 25, "Group A")
        st.rerun()

    if preset_col2.button("⚡ Load Intermediate Example"):
        load_preset("Maya Lin", "STU-3002", 58, 62, 60, 55, 59, 57, "Group B")
        st.rerun()

    if preset_col3.button("🚀 Load Advanced Example"):
        load_preset("Rohan Verma", "STU-3003", 75, 78, 70, 74, 76, 72, "Group A")
        st.rerun()

    if preset_col4.button("🌟 Load Expert Example"):
        load_preset("Elena Rostova", "STU-3004", 94, 92, 85, 90, 95, 96, "Group B")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("assessment_form"):
        col_id, col_name, col_grp = st.columns([1, 2, 1])
        with col_id:
            s_id = st.text_input("Student ID", value=st.session_state.current_student["student_id"])
        with col_name:
            s_name = st.text_input("Student Full Name", value=st.session_state.current_student["student_name"])
        with col_grp:
            current_grp = st.session_state.current_student["assessment_group"]
            s_group = st.selectbox(
                "Assessment Group",
                ["Group A", "Group B"],
                index=0 if current_grp == "Group A" else 1,
                help="Demographic or cohort group used solely for fairness monitoring."
            )

        st.markdown("#### Assessment Competency Scores (0 - 100)")
        sc = st.session_state.current_student["scores"]

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            prog = st.slider("Programming Skills", 0.0, 100.0, float(sc["programming"]), step=1.0)
            prob = st.slider("Problem-Solving Skills", 0.0, 100.0, float(sc["problem_solving"]), step=1.0)
            comm = st.slider("Communication Skills", 0.0, 100.0, float(sc["communication"]), step=1.0)

        with f_col2:
            logic = st.slider("Logical Reasoning", 0.0, 100.0, float(sc["logical_reasoning"]), step=1.0)
            tech = st.slider("Technical Knowledge", 0.0, 100.0, float(sc["technical_knowledge"]), step=1.0)
            prac = st.slider("Practical Application", 0.0, 100.0, float(sc["practical_application"]), step=1.0)

        submitted = st.form_submit_button("🚀 Run AI Evaluation & Analysis", use_container_width=True)

        if submitted:
            # Input validation
            if not s_id.strip() or not s_name.strip():
                st.error("Please provide both Student ID and Student Name.")
            else:
                updated_scores = {
                    "programming": prog,
                    "problem_solving": prob,
                    "communication": comm,
                    "logical_reasoning": logic,
                    "technical_knowledge": tech,
                    "practical_application": prac
                }

                # Compute prediction & explanation
                pred_res = predict_student_skill(model, updated_scores)
                exp_res = explain_prediction(
                    updated_scores,
                    pred_res["predicted_category"],
                    metrics["feature_importances"],
                    cohort_df
                )

                # Update session state
                st.session_state.current_student["student_id"] = s_id.strip()
                st.session_state.current_student["student_name"] = s_name.strip()
                st.session_state.current_student["scores"] = updated_scores
                st.session_state.current_student["assessment_group"] = s_group
                st.session_state.current_student["prediction_result"] = pred_res
                st.session_state.current_student["explanation"] = exp_res

                st.success(
                    f"✅ Evaluation computed successfully for {s_name}! "
                    f"Predicted Skill Level: **{pred_res['predicted_category']}** "
                    f"({pred_res['confidence']*100:.1f}% confidence)."
                )


# ---------------------------------------------------------
# PAGE 3: AI PREDICTION
# ---------------------------------------------------------
elif selected_page == "3. AI Prediction":
    st.title("🤖 Supervised Classification: AI Prediction")
    st.markdown(
        "This module utilizes a **Random Forest Classifier** trained on objective assessment metrics "
        "to categorize the student into one of four skill categories: **Beginner**, **Intermediate**, **Advanced**, or **Expert**."
    )

    stu = st.session_state.current_student
    if stu["prediction_result"] is None:
        # Compute on the fly if not yet run
        stu["prediction_result"] = predict_student_skill(model, stu["scores"])
        stu["explanation"] = explain_prediction(
            stu["scores"],
            stu["prediction_result"]["predicted_category"],
            metrics["feature_importances"],
            cohort_df
        )

    pred = stu["prediction_result"]
    category = pred["predicted_category"]
    confidence = pred["confidence"]
    probs = pred["class_probabilities"]

    # Student Summary Banner
    st.markdown(f"""
    <div style="background:#f1f5f9; padding:16px 20px; border-radius:10px; margin-bottom:20px;">
        <span style="font-size:1.15rem; font-weight:700; color:#0f172a;">Student: {stu['student_name']}</span>
        <span style="color:#64748b; margin-left:12px;">(ID: {stu['student_id']} | Cohort: {stu['assessment_group']})</span>
    </div>
    """, unsafe_allow_html=True)

    # Prediction Card & Confidence
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.markdown("### AI Classification Result")
        st.markdown(f"#### Assigned Category: {get_skill_badge_html(category)}", unsafe_allow_html=True)
        st.metric(label="Model Confidence Score", value=f"{confidence * 100:.1f}%")
        st.progress(float(confidence))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Competency Scores Summary")
        scores_df = pd.DataFrame([stu["scores"]]).T.reset_index()
        scores_df.columns = ["Competency", "Score"]
        scores_df["Competency"] = scores_df["Competency"].apply(lambda x: x.replace("_", " ").title())
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

    with p_col2:
        st.markdown("### Class Probability Distribution")
        prob_df = pd.DataFrame([
            {"Category": cat, "Probability": probs.get(cat, 0.0) * 100}
            for cat in CATEGORIES
        ])

        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ["#3b82f6" if c == category else "#cbd5e1" for c in CATEGORIES]
        bars = ax.bar(prob_df["Category"], prob_df["Probability"], color=bar_colors, width=0.5)
        ax.set_ylabel("Probability (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 📈 Model Evaluation Metrics (Test Split)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m3.metric("Recall", f"{metrics['recall']*100:.1f}%")
    m4.metric("F1-Score", f"{metrics['f1_score']*100:.1f}%")


# ---------------------------------------------------------
# PAGE 4: EXPLAINABLE AI (XAI)
# ---------------------------------------------------------
elif selected_page == "4. Explainable AI":
    st.title("🔍 Explainable AI (XAI)")
    st.markdown(
        "Explainable AI ensures that machine learning decisions are not treated as an uninterpretable "
        "'black box'. This page highlights both **global model feature importance** and **local explanations** for the active student."
    )

    stu = st.session_state.current_student
    if stu["explanation"] is None or stu["prediction_result"] is None:
        stu["prediction_result"] = predict_student_skill(model, stu["scores"])
        stu["explanation"] = explain_prediction(
            stu["scores"],
            stu["prediction_result"]["predicted_category"],
            metrics["feature_importances"],
            cohort_df
        )

    exp = stu["explanation"]
    pred = stu["prediction_result"]

    # Natural Language Summary Card
    st.markdown(f"""
    <div style="background:#eff6ff; border-left: 4px solid #2563eb; padding:16px 20px; border-radius:0 10px 10px 0; margin-bottom:20px;">
        <h4 style="margin-top:0; color:#1e40af;">💡 Decision Interpretation for {stu['student_name']}</h4>
        <p style="font-size:1.02rem; color:#1e293b; margin-bottom:6px;">{exp['text_summary']}</p>
        <small style="color:#64748b;">Predicted Category: <b>{pred['predicted_category']}</b> | Confidence: <b>{pred['confidence']*100:.1f}%</b></small>
    </div>
    """, unsafe_allow_html=True)

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.subheader("Global Feature Importance (Overall Model)")
        st.caption("Shows which competencies influence the AI classifier most heavily across all students.")

        fi_sorted = sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)
        fi_labels = [k.replace("_", " ").title() for k, _ in fi_sorted]
        fi_vals = [v * 100 for _, v in fi_sorted]

        fig, ax = plt.subplots(figsize=(6, 4.2))
        bars = ax.barh(fi_labels[::-1], fi_vals[::-1], color="#0284c7", height=0.55)
        ax.set_xlabel("Importance Weight (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1f}%",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    with xai_col2:
        st.subheader("Student Scores vs. Cohort Benchmarks")
        st.caption(f"Comparison of {stu['student_name']}'s scores against average student population scores.")

        labels = [f.replace("_", " ").title() for f in FEATURE_COLUMNS]
        stu_vals = [stu["scores"][f] for f in FEATURE_COLUMNS]
        avg_vals = [exp["cohort_averages"][f] for f in FEATURE_COLUMNS]

        x = np.arange(len(labels))
        width = 0.35

        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        ax2.bar(x - width/2, stu_vals, width, label=stu['student_name'], color="#6366f1")
        ax2.bar(x + width/2, avg_vals, width, label="Cohort Average", color="#94a3b8")
        ax2.set_ylabel("Score (0 - 100)", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax2.set_ylim(0, 105)
        ax2.legend(frameon=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    f_lead1, f_lead2 = st.columns(2)
    with f_lead1:
        st.markdown("#### 🌟 Top Contributing Strengths")
        if exp["top_strengths"]:
            for s in exp["top_strengths"]:
                score = stu["scores"][s]
                st.markdown(f"- **{s.replace('_', ' ').title()}**: `{score:.1f}/100` (+{exp['score_differences'][s]:.1f} above cohort average)")
        else:
            st.markdown("*Scores are uniformly distributed across foundational baseline.*")

    with f_lead2:
        st.markdown("#### 🎯 Recommended Focus & Growth Areas")
        if exp["growth_areas"]:
            for g in exp["growth_areas"]:
                score = stu["scores"][g]
                diff = exp['score_differences'][g]
                st.markdown(f"- **{g.replace('_', ' ').title()}**: `{score:.1f}/100` ({diff:+.1f} vs cohort average)")
        else:
            st.markdown("*Exceptional performance across all assessed dimensions.*")


# ---------------------------------------------------------
# PAGE 5: FAIRNESS ANALYSIS
# ---------------------------------------------------------
elif selected_page == "5. Fairness Analysis":
    st.title("⚖️ Algorithmic Fairness & Disparity Evaluation")
    st.markdown(
        "An AI system should not systematically disadvantage students from any background, cohort, or institution. "
        "Here, we analyze group-level accuracy and demographic parity between **Group A** and **Group B**."
    )

    # Prepare predictions for the full synthetic test/cohort set
    eval_df = cohort_df.copy()
    eval_df["predicted_skill"] = model.predict(eval_df[FEATURE_COLUMNS])
    eval_df["actual_skill"] = eval_df["skill_level"]

    fairness_res = evaluate_fairness_metrics(eval_df, fairness_tolerance=0.10)
    group_stats = fairness_res["group_stats"]
    disparities = fairness_res["disparities"]

    # Fairness Alert Status Banner
    if fairness_res["is_fair"] and len(fairness_res["warnings"]) == 0:
        st.success(
            "✅ **Fairness Check Passed**: Group accuracy and demographic selection rates are well within the "
            "allowable tolerance threshold (10%). No systemic bias detected."
        )
    else:
        st.warning("⚠️ **Fairness Alert Detected:**")
        for w in fairness_res["warnings"]:
            st.markdown(f"- {w}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparative Metric Cards
    c1, c2, c3 = st.columns(3)
    groups_list = list(group_stats.keys())

    if len(groups_list) >= 2:
        g1, g2 = groups_list[0], groups_list[1]
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">{g1} Accuracy</div>
                <div class="metric-card-value">{group_stats[g1]['accuracy']*100:.1f}%</div>
                <div class="metric-card-sub">Sample size: {group_stats[g1]['count']} students</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">{g2} Accuracy</div>
                <div class="metric-card-value">{group_stats[g2]['accuracy']*100:.1f}%</div>
                <div class="metric-card-sub">Sample size: {group_stats[g2]['count']} students</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            diff = disparities.get("accuracy_difference", 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Accuracy Gap</div>
                <div class="metric-card-value">{diff*100:.1f}%</div>
                <div class="metric-card-sub">{"Within Tolerance" if diff <= 0.10 else "High Disparity"}</div>
            </div>
            """, unsafe_allow_html=True)

    # Visual Group Comparison
    f_chart1, f_chart2 = st.columns(2)

    with f_chart1:
        st.subheader("Group-Wise Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        grp_names = list(group_stats.keys())
        grp_accs = [group_stats[g]["accuracy"] * 100 for g in grp_names]

        bars = ax.bar(grp_names, grp_accs, color=["#3b82f6", "#10b981"], width=0.45)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% Benchmark")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="lower right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    with f_chart2:
        st.subheader("Skill Level Distribution by Group")
        dist_data = []
        for g in grp_names:
            for cat in CATEGORIES:
                dist_data.append({
                    "Group": g,
                    "Category": cat,
                    "Percentage": group_stats[g]["distribution"].get(cat, 0.0) * 100
                })
        dist_df = pd.DataFrame(dist_data)

        # Plot grouped bars
        x = np.arange(len(CATEGORIES))
        width = 0.35
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        vals1 = dist_df[dist_df["Group"] == grp_names[0]]["Percentage"].values
        vals2 = dist_df[dist_df["Group"] == grp_names[1]]["Percentage"].values

        ax2.bar(x - width/2, vals1, width, label=grp_names[0], color="#3b82f6")
        ax2.bar(x + width/2, vals2, width, label=grp_names[1], color="#10b981")
        ax2.set_ylabel("Share of Group (%)", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(CATEGORIES, fontsize=9)
        ax2.set_ylim(0, 50)
        ax2.legend(frameon=False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown(get_fairness_educational_summary())


# ---------------------------------------------------------
# PAGE 6: HUMAN REVIEW & COLLABORATION
# ---------------------------------------------------------
elif selected_page == "6. Human Review":
    st.title("✍️ Human-AI Collaboration: Evaluator Review")
    st.markdown(
        "In this decision-support workflow, the **AI acts as an advisory assistant**, not the ultimate judge. "
        "A human educator reviews the AI's classification, examines borderline scores or contextual notes, "
        "and either confirms or overrides the prediction before saving the record into the permanent SQLite audit log."
    )

    stu = st.session_state.current_student
    if stu["prediction_result"] is None:
        stu["prediction_result"] = predict_student_skill(model, stu["scores"])
        stu["explanation"] = explain_prediction(
            stu["scores"],
            stu["prediction_result"]["predicted_category"],
            metrics["feature_importances"],
            cohort_df
        )

    pred = stu["prediction_result"]
    category = pred["predicted_category"]
    confidence = pred["confidence"]

    # Review Container
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Student Profile & AI Recommendation")
        st.write(f"**Student ID:** `{stu['student_id']}`")
        st.write(f"**Student Name:** {stu['student_name']}")
        st.write(f"**Assessment Cohort:** {stu['assessment_group']}")
        st.markdown(f"**AI Suggested Skill Level:** {get_skill_badge_html(category)}", unsafe_allow_html=True)
        st.write(f"**AI Confidence:** `{confidence * 100:.1f}%`")

        st.markdown("##### Candidate Scorecard:")
        score_items = [f"**{k.replace('_', ' ').title()}:** {v:.1f}" for k, v in stu["scores"].items()]
        st.markdown(" • " + "  \n • ".join(score_items))

    with col_right:
        st.subheader("🧑‍🏫 Evaluator Action Panel")

        with st.form("human_review_form"):
            decision_action = st.radio(
                "Review Decision",
                ["Accept AI Prediction", "Override AI Prediction"],
                index=0
            )

            final_level = category
            if decision_action == "Override AI Prediction":
                final_level = st.selectbox(
                    "Select Corrected Skill Level",
                    CATEGORIES,
                    index=CATEGORIES.index(category)
                )

            evaluator_name = st.text_input("Evaluator Name / ID", value="Dr. Vance (Course Instructor)")
            evaluator_notes = st.text_area(
                "Review Comments / Justification",
                placeholder="Enter observations, special coursework considerations, or justification for override...",
                value="Assessment reviewed. Technical scores strongly support this classification."
            )

            submit_review = st.form_submit_button("💾 Finalize & Save Evaluation to Database", use_container_width=True)

            if submit_review:
                review_status = "AI Accepted" if decision_action == "Accept AI Prediction" else "Human Overridden"

                row_id = save_assessment(
                    student_id=stu["student_id"],
                    student_name=stu["student_name"],
                    programming=stu["scores"]["programming"],
                    problem_solving=stu["scores"]["problem_solving"],
                    communication=stu["scores"]["communication"],
                    logical_reasoning=stu["scores"]["logical_reasoning"],
                    technical_knowledge=stu["scores"]["technical_knowledge"],
                    practical_application=stu["scores"]["practical_application"],
                    assessment_group=stu["assessment_group"],
                    ai_prediction=category,
                    ai_confidence=confidence,
                    final_decision=final_level,
                    review_status=review_status,
                    evaluator_name=evaluator_name,
                    evaluator_notes=evaluator_notes
                )

                st.success(
                    f"🎉 Assessment successfully finalized and saved to SQLite! (Record #{row_id})\n\n"
                    f"**Final Decision:** {final_level} ({review_status})"
                )


# ---------------------------------------------------------
# PAGE 7: ASSESSMENT HISTORY
# ---------------------------------------------------------
elif selected_page == "7. Assessment History":
    st.title("📜 Assessment History & Audit Log")
    st.markdown(
        "All completed evaluations and human reviews are stored persistently in the **SQLite database (`assessments.db`)**. "
        "You can inspect past records, track review overrides, and export data for institutional reporting."
    )

    df_history = load_all_assessments()

    if df_history.empty:
        st.info("No assessments recorded in SQLite yet. Use the 'Human Review' page to finalize your first evaluation!")
    else:
        # Filter controls
        f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
        with f_c1:
            search_query = st.text_input("🔍 Search by Student Name or ID", "")
        with f_c2:
            status_filter = st.selectbox("Filter by Status", ["All", "AI Accepted", "Human Overridden"])
        with f_c3:
            skill_filter = st.selectbox("Filter by Final Skill", ["All"] + CATEGORIES)

        filtered_df = df_history.copy()

        if search_query.strip():
            query = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["student_name"].str.lower().str.contains(query) |
                filtered_df["student_id"].str.lower().str.contains(query)
            ]

        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["review_status"] == status_filter]

        if skill_filter != "All":
            filtered_df = filtered_df[filtered_df["final_decision"] == skill_filter]

        st.caption(f"Showing **{len(filtered_df)}** of **{len(df_history)}** total records.")

        # Display Table
        display_cols = [
            "id", "student_id", "student_name", "assessment_group",
            "ai_prediction", "final_decision", "review_status",
            "evaluator_name", "created_at"
        ]

        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Download button for CSV
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered History as CSV",
            data=csv_data,
            file_name="student_assessment_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Quick breakdown
        st.markdown("---")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown("#### Review Status Breakdown")
            status_counts = df_history["review_status"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
                   colors=["#10b981", "#f59e0b"], startangle=90)
            fig.tight_layout()
            st.pyplot(fig)

        with h_col2:
            st.markdown("#### Final Approved Skill Levels")
            final_counts = df_history["final_decision"].value_counts().reindex(CATEGORIES, fill_value=0)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(final_counts.index, final_counts.values, color=["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"], width=0.5)
            ax2.set_ylabel("Count", fontsize=9)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            fig2.tight_layout()
            st.pyplot(fig2)

