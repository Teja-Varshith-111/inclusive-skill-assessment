# Comprehensive Academic Project Report
## Inclusive Assessment and Skill-Evaluation Decision Support System

**Course / Subject:** Artificial Intelligence & Machine Learning Project  
**Target Level:** Undergraduate / College Capstone / Lab Project  
**Technologies:** Python, Scikit-learn, Streamlit, Pandas, NumPy, Matplotlib, SQLite  

---

## Abstract

In contemporary higher education and technical skill evaluations, automated grading tools frequently operate as opaque "black boxes" that obscure decision rationale and perpetuate demographic or cohort-level disparities. This project presents the **Inclusive Assessment and Skill-Evaluation Decision Support System**, an open, transparent, and ethical machine learning platform designed to evaluate student competencies across six core technical dimensions: Programming, Problem-Solving, Communication, Logical Reasoning, Technical Knowledge, and Practical Application. 

The system implements four interrelated foundational Artificial Intelligence pillars:
1. **Supervised Classification:** An ensemble Random Forest model categorizing students into four structured skill tiers (`Beginner`, `Intermediate`, `Advanced`, `Expert`) with **91.88% test accuracy**. Sensitive demographic attributes (names, IDs, cohorts) are strictly omitted from the feature matrix to enforce demographic neutrality.
2. **Explainable AI (XAI):** Global Mean Decrease in Impurity (Gini Importance) and localized score-versus-benchmark comparisons provide plain-English explanations of classification drivers for each student.
3. **Fairness Evaluation:** Algorithmic bias audits measure demographic parity and group accuracy disparity between demographic cohorts, triggering active warning alerts when performance discrepancies exceed established tolerance thresholds ($10\%$).
4. **Human-AI Collaboration:** The model operates strictly in an advisory capacity within a Decision Support System (DSS) framework. Human evaluators retain full authority to confirm or override predictions, append qualitative justification notes, and commit immutable audit records into an embedded **SQLite** database.

Extensive automated and manual quality assurance testing across 64 individual test scenarios yielded a **100% pass rate**, proving system robustness, fault tolerance, and practical readiness for academic deployment.

---

## Chapter 1: Introduction

### 1.1 Background
Evaluation of technical skill sets among engineering students has traditionally relied on subjective manual scoring or simplistic cutoffs that fail to capture multidimensional competency. The recent introduction of automated artificial intelligence tools has accelerated scoring throughput but introduced critical systemic risks: algorithmic bias, opaque decision-making, and the removal of human oversight in high-stakes academic determinations.

### 1.2 Problem Statement
Existing AI-based evaluation systems suffer from three primary vulnerabilities:
- **Opacity (The Black-Box Problem):** Students and faculty receive automated classifications without insight into which competencies drove the assessment.
- **Algorithmic Disparity:** Systems trained on skewed cohorts can produce unequal accuracy or selection rates across different student groups, exacerbating systemic inequities.
- **Autonomous Overreach:** Fully autonomous systems strip educators of pedagogical context, disallowing humane overrides for extenuating circumstances or exceptional portfolio demonstrations.

### 1.3 Project Objectives
The primary goal of this project is to develop a beginner-friendly, fully functional Decision Support System that:
1. Classifies student competencies into four intuitive tiers with greater than 85% accuracy.
2. Excludes non-competency metadata (student names and IDs) from model inference.
3. Generates transparent global and local explanations for every classification.
4. Monitors group-wise parity metrics in real time to prevent demographic bias.
5. Empowers educators to review, accept, or override AI outputs with mandatory audit logs stored in a local SQLite database.
6. Delivers an intuitive 7-page Streamlit web interface suitable for classroom demonstration.

---

## Chapter 2: Theoretical Foundations & AI Concepts

### 2.1 Supervised Classification via Random Forests
Supervised machine learning trains an algorithm on a feature matrix $X \in \mathbb{R}^{n \times d}$ and known labels $y \in \{C_1, C_2, C_3, C_4\}$. While single decision trees suffer from high variance and prone to overfitting, Random Forest applies Bootstrap Aggregation (bagging) across $B = 100$ independent decision trees. Node splitting is governed by minimizing Gini Impurity:

$$I_G(p) = 1 - \sum_{i=1}^{J} p_i^2$$

The final class prediction is determined by majority voting over tree outputs:

$$\hat{y} = \arg\max_{c} \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(T_b(x) = c)$$

### 2.2 Explainable AI (XAI) Architecture
Explainability is achieved through two complementary mechanisms:
1. **Global Feature Importance:** Measures the normalized total reduction of Gini impurity brought by each feature across all trees in the ensemble.
2. **Local Counterfactual Comparison:** Contrasts an individual student's score vector $x^{(i)}$ against cohort benchmark means $\bar{x}$, identifying primary drivers where $(x_j^{(i)} - \bar{x}_j) \cdot W_j$ is maximized.

### 2.3 Algorithmic Fairness in Educational Measurement
Fairness auditing evaluates whether the model operates equitably across sensitive groups $G \in \{A, B\}$:
- **Demographic Parity:** Ensures that selection into advanced tiers is independent of group membership:
  $$|P(\hat{Y} \in \{\text{Adv, Exp}\} \mid G=A) - P(\hat{Y} \in \{\text{Adv, Exp}\} \mid G=B)| \le \epsilon$$
- **Equal Accuracy:** Ensures that error rates do not disproportionately affect one demographic group:
  $$|\text{Accuracy}(G=A) - \text{Accuracy}(G=B)| \le \tau \quad (\tau = 0.10)$$

### 2.4 Human-AI Decision Support Paradigm
Rather than replacing educators, the system operates as a Decision Support System (DSS). The human educator retains primary agency, receiving AI recommendations as secondary advisory inputs. Overrides are formally logged, creating an immutable pedagogical record.

---

## Chapter 3: System Architecture & Design

### 3.1 Architectural Diagram

```text
[Student Assessment Input Form]
               │
               ▼
   [Feature Preprocessing] ──(Omit Name/ID)──► [Random Forest Classifier]
                                                       │
                                   ┌───────────────────┴───────────────────┐
                                   ▼                                       ▼
                       [Explainable AI Engine]                  [Fairness Audit Engine]
                       (Local & Global Drivers)                (Cohort Parity & Alerts)
                                   │                                       │
                                   └───────────────────┬───────────────────┘
                                                       │
                                                       ▼
                                         [Human Evaluator Action Panel]
                                           (Accept / Override / Notes)
                                                       │
                                                       ▼
                                         [SQLite Relational Database]
                                          (Immutable Audit History)
```

### 3.2 Database Schema (`assessments.db`)

| Column Name | Data Type | Constraint | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique evaluation identifier |
| `student_id` | TEXT | NOT NULL | Student institutional identification number |
| `student_name` | TEXT | NOT NULL | Student full legal or preferred name |
| `programming` | REAL | NOT NULL | Programming competency score (0.0–100.0) |
| `problem_solving` | REAL | NOT NULL | Problem-solving competency score (0.0–100.0) |
| `communication` | REAL | NOT NULL | Communication competency score (0.0–100.0) |
| `logical_reasoning` | REAL | NOT NULL | Logical reasoning competency score (0.0–100.0) |
| `technical_knowledge` | REAL | NOT NULL | Technical knowledge competency score (0.0–100.0) |
| `practical_application` | REAL | NOT NULL | Practical application competency score (0.0–100.0) |
| `assessment_group` | TEXT | NOT NULL | Demographic/cohort cohort tag (`Group A` / `Group B`) |
| `ai_prediction` | TEXT | NOT NULL | Skill tier predicted by machine learning model |
| `ai_confidence` | REAL | NOT NULL | Maximum predicted class probability (0.0–1.0) |
| `final_decision` | TEXT | NOT NULL | Final approved skill tier (AI or human overridden) |
| `review_status` | TEXT | NOT NULL | `'AI Accepted'` or `'Human Overridden'` |
| `evaluator_name` | TEXT | DEFAULT 'Anonymous' | Name of educator conducting review |
| `evaluator_notes` | TEXT | DEFAULT 'None' | Qualitative justification notes |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Audit timestamp |

---

## Chapter 4: Implementation Details

### 4.1 Synthetic Cohort Generation
In `dataset.py`, 800 synthetic records are synthesized across four population distributions with Gaussian noise $\mathcal{N}(0, 2.5)$. Ground truth skill tiers are computed using weighted linear combination:

$$\text{Composite} = 0.22 \times \text{Prog} + 0.20 \times \text{Prob} + 0.16 \times \text{Logic} + 0.15 \times \text{Tech} + 0.15 \times \text{Prac} + 0.12 \times \text{Comm}$$

Tiers are assigned as:
- $\text{Composite} < 45 \implies \text{Beginner}$
- $45 \le \text{Composite} < 65 \implies \text{Intermediate}$
- $65 \le \text{Composite} < 82 \implies \text{Advanced}$
- $\text{Composite} \ge 82 \implies \text{Expert}$

### 4.2 Supervised Learning Implementation
In `model.py`, the dataset is partitioned via an 80/20 stratified split. Scikit-learn's `RandomForestClassifier` is instantiated with `n_estimators=100`, `max_depth=6`, and `random_state=42`. Stratification preserves identical class balances in both splits.

---

## Chapter 5: Results & Discussion

### 5.1 Model Evaluation Metrics
Tested on 160 unseen student test instances:
- **Overall Accuracy:** $91.88\%$
- **Weighted Precision:** $92.00\%$
- **Weighted Recall:** $91.88\%$
- **Weighted F1-Score:** $91.88\%$

### 5.2 Feature Importance Weights
- Problem Solving: $21.43\%$
- Practical Application: $20.71\%$
- Programming: $19.86\%$
- Logical Reasoning: $17.71\%$
- Technical Knowledge: $15.17\%$
- Communication: $5.13\%$

### 5.3 Fairness Evaluation Findings
- **Group A Accuracy:** $91.95\%$ ($n=87$)
- **Group B Accuracy:** $91.78\%$ ($n=73$)
- **Accuracy Disparity:** $|91.95\% - 91.78\%| = 0.17\%$ (Well below the $10\%$ threshold)
- **High-Skill Selection Parity:** Group A: $43.68\%$, Group B: $42.47\%$ (Disparity = $1.21\%$)

---

## Chapter 6: Quality Assurance & Testing

A complete QA testing campaign was conducted covering 64 test cases:
- **Unit & Integration Suite (`tests/`):** 24 automated test cases covering input boundaries, ML probabilities, fairness thresholds, database CRUD operations, and end-to-end integration. (Execution time: 1.226s, 100% pass rate).
- **Master Test Harness (`run_all_test_cases.py`):** 40 functional, negative, boundary, and UI test cases verified with zero regressions.

---

## Chapter 7: Conclusion & Future Scope

### 7.1 Conclusion
The **Inclusive Assessment and Skill-Evaluation Decision Support System** establishes a reproducible blueprint for ethical, explainable, and fair AI applications in education. By coupling high classification accuracy ($91.88\%$) with real-time bias audits and human-in-the-loop governance, the project ensures that technological automation elevates rather than compromises pedagogical integrity.

### 7.2 Future Scope
1. **Adaptive Testing Integration:** Integrating dynamic difficulty adjustment based on real-time student responses.
2. **Multi-Evaluator Consensus:** Allowing multiple committee members to register simultaneous review scores.
3. **Continuous Active Retraining:** Automatically queuing human-overridden edge cases for scheduled model recalibration.

