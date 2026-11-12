# Inclusive Assessment and Skill-Evaluation Decision Support System

An educational, beginner-friendly AI/ML Decision Support System built in Python and Streamlit to evaluate student skills, classify their competency levels, explain AI recommendations, evaluate algorithmic fairness across cohorts, and empower human educators to review and finalize decisions.

---

## 🎯 Project Objective

To build an AI-assisted evaluation system that:
1. Assesses student scores across **6 key technical competencies** (Programming, Problem Solving, Communication, Logical Reasoning, Technical Knowledge, and Practical Application).
2. Classifies students into **4 skill tiers**: `Beginner`, `Intermediate`, `Advanced`, and `Expert`.
3. Explains AI decisions using **Explainable AI (XAI)** techniques.
4. Audits algorithmic bias between assessment cohorts via **Fairness Evaluation**.
5. Implements **Human-AI Collaboration** where teachers/evaluators have full authority to confirm or override predictions before saving them to a **SQLite** database.

---

## 🧠 4 Core AI Concepts Explained

| AI Concept | How It Is Implemented in This Project | Where to Find It |
| :--- | :--- | :--- |
| **1. Supervised Classification** | Trains a **Random Forest Classifier** (`scikit-learn`) on objective skill scores. Evaluates performance using Accuracy, Precision, Recall, and F1-Score. Explicitly omits student identifiers from features to prevent bias. | `model.py` & Page 3 (`AI Prediction`) |
| **2. Explainable AI (XAI)** | Computes **global feature importance** (which skills drive decisions across all students) and **local explanations** (comparing an individual student's scores against cohort benchmarks with plain-English insights). | `model.py` & Page 4 (`Explainable AI`) |
| **3. Fairness Evaluation** | Computes group-level accuracy and demographic selection rates between cohorts (`Group A` vs `Group B`). Flags automatic warnings if disparities exceed tolerance thresholds (e.g. 10%). | `fairness.py` & Page 5 (`Fairness Analysis`) |
| **4. Human-AI Collaboration** | The AI serves as an advisory assistant. Human evaluators review AI predictions alongside student scorecards, with full power to accept or override the level, add commentary, and record an audit log in SQLite. | `database.py` & Page 6 (`Human Review`) |

---

## 📁 Project Structure

```text
c:\Users\tejav\AI Project\
│
├── app.py                  # Main Streamlit web application (7 interactive pages)
├── dataset.py              # Synthetic dataset generator (800 student records)
├── model.py                # Scikit-learn model training, prediction & XAI logic
├── fairness.py             # Fairness evaluation and disparity detection engine
├── database.py             # SQLite database management (audit log & history)
├── requirements.txt        # Python package dependencies
├── students_dataset.csv    # Generated synthetic student assessment dataset
├── assessments.db          # SQLite database storing human-reviewed assessments
├── skill_classifier.pkl    # Serialized trained Random Forest model & metrics
└── README.md               # Complete project documentation & guide
```

---

## 💻 Technologies Used

- **Python 3.10+ / 3.13**
- **Streamlit**: Modern interactive web interface
- **Scikit-learn**: Random Forest classification & evaluation metrics
- **Pandas & NumPy**: Data processing and numeric calculations
- **Matplotlib**: Clean visual charts and distributions
- **SQLite**: Local relational database for persistent storage

---

## 🚀 How to Run on Windows

### 1. Open PowerShell / Command Prompt
Open your terminal in this directory:
```powershell
cd "c:\Users\tejav\AI Project"
```

### 2. Verify / Install Dependencies
If needed, install dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Re-generate Dataset or Retrain Model
You can run any module standalone:
```powershell
# Generate fresh synthetic dataset:
python dataset.py

# Train and test the Random Forest model:
python model.py

# Check fairness metrics:
python fairness.py
```

### 4. Launch the Streamlit Web Application
Run:
```powershell
streamlit run app.py
```
Streamlit will automatically open your default web browser at `http://localhost:8501`.

---

## 🖥️ Application Pages Overview

1. **📊 Dashboard**: High-level overview of total finalized evaluations, model accuracy, average competency scores, and skill distributions.
2. **📝 Student Assessment**: Intuitive input form with sliders for each skill (0–100) and preset example buttons (`Beginner`, `Intermediate`, `Advanced`, `Expert`).
3. **🤖 AI Prediction**: Displays the predicted category, model confidence percentage, and class probability distribution.
4. **🔍 Explainable AI**: Visualizes global feature importance and displays personalized score-vs-benchmark charts with clear English explanations.
5. **⚖️ Fairness Analysis**: Analyzes accuracy and high-tier representation between `Group A` and `Group B` with automatic disparity alerts.
6. **✍️ Human Review**: Human-in-the-loop action panel allowing evaluators to accept or override AI predictions and record audit notes.
7. **📜 Assessment History**: Searchable SQLite table with filtering, status distribution charts, and one-click CSV export.

---

## 🎓 College-Level Demonstration Script

When demonstrating this project to instructors or peers:
1. **Show the Problem**: Traditional grading can be subjective; automated AI models can be opaque ("black box") or biased.
2. **Show the Dataset (`dataset.py`)**: Explain how synthetic data simulates 800 students without privacy issues, and why student name/ID are never used as model inputs.
3. **Run a Prediction (`app.py` -> Student Assessment)**: Click "Load Advanced Example" or input custom numbers. Show the AI Prediction and confidence score.
4. **Demonstrate Explainability (`Explainable AI`)**: Point out which skills carried the most weight and why this student qualified for that level.
5. **Demonstrate Fairness (`Fairness Analysis`)**: Show how the system audits accuracy across groups to detect and prevent demographic bias.
6. **Demonstrate Human Agency (`Human Review`)**: Override a borderline case and explain why human oversight is essential in high-stakes educational AI.
7. **Show the Audit Trail (`Assessment History`)**: Show the record saved into `assessments.db` with the evaluator's justification notes.

