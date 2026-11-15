# Inclusive Assessment and Skill-Evaluation Decision Support System

An AI-powered skill assessment platform designed to evaluate student competencies, provide explainable assessment results, analyse group-level fairness and support human-in-the-loop evaluation.

<p align="center">
  <a href="https://inclusive-skill-assessment.streamlit.app/">
    <strong>Live Application</strong>
  </a>
  |
  <a href="https://github.com/Teja-Varshith-111/inclusive-skill-assessment">
    <strong>GitHub Repository</strong>
  </a>
</p>

---

## 1. Project Information

| Field | Details |
|---|---|
| Project Title | Inclusive Assessment and Skill-Evaluation Decision Support System |
| Repository Name | inclusive-skill-assessment |
| Problem Statement Source | Smart India Hackathon (SIH) 2024 |
| Mapped ID | SIH1777 |
| Application Type | AI/ML-based Web Application |
| Deployment Platform | Streamlit Cloud |
| Machine Learning Algorithm | Random Forest Classifier |
| Database | SQLite |

### Team Members

| Name | Registration Number |
|---|---|
| Akula Teja Varshith | 12510396 |
| Pullannagari Sai Vishaal Mohan | 12519729 |

---

## 2. Introduction

Traditional student assessment methods may not always provide a comprehensive understanding of individual skills and competencies. They can also make it difficult for educators to understand the reasons behind assessment outcomes or identify differences in assessment patterns across student groups.

The **Inclusive Assessment and Skill-Evaluation Decision Support System** addresses this challenge through an AI-assisted assessment platform.

The application evaluates student competencies using a machine learning classification model, provides interpretable assessment insights, analyses group-level fairness and supports human review of generated predictions.

The system is designed as a decision-support tool. Its predictions are intended to assist educators and evaluators rather than replace human judgement.

---

## 3. Project Objectives

The major objectives are:

1. Develop an AI-powered platform for student skill assessment.
2. Evaluate multiple competency areas using structured assessment inputs.
3. Classify assessment outcomes into different proficiency levels.
4. Provide explainable insights into machine learning predictions.
5. Analyse differences in assessment outcomes between groups.
6. Support human review and correction of automated predictions.
7. Maintain assessment records for future reference and analysis.

---

## 4. Key Features

### 4.1 Multi-Dimensional Skill Assessment

The platform evaluates six competency areas:

- Programming
- Problem Solving
- Communication
- Logical Reasoning
- Technical Knowledge
- Practical Application

### 4.2 Machine Learning-Based Classification

A Random Forest classification model analyses assessment inputs and predicts a student's proficiency category.

The implemented categories are:

- Beginner
- Intermediate
- Advanced
- Expert

### 4.3 Explainable AI

The system provides interpretable insights into assessment predictions through:

- Feature-importance analysis.
- Identification of influential assessment features.
- Comparison of individual assessment results with cohort information.

### 4.4 Fairness Analysis

The application includes group-level fairness analysis to examine differences in assessment outcomes between Group A and Group B.

This functionality helps evaluators identify potential differences in model behaviour that may require further investigation.

### 4.5 Human-in-the-Loop Evaluation

Human review functionality allows an evaluator to review and override an automated prediction when necessary.

This helps maintain human oversight in the assessment process.

### 4.6 Assessment History

The application uses an SQLite database to maintain assessment-related records and support historical review.

---

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | Interactive web application |
| Scikit-learn | Machine learning model development |
| Random Forest | Skill classification |
| Pandas | Dataset handling and processing |
| NumPy | Numerical operations |
| SQLite | Assessment data storage |
| Pickle | Model serialization |
| Git and GitHub | Version control and source code management |
| Streamlit Cloud | Application deployment |

---

## 6. Dataset Details

The project uses a synthetic assessment dataset for model development and demonstration.

| Dataset Attribute | Details |
|---|---|
| Dataset Source | Project-generated synthetic dataset |
| Dataset File | students_dataset.csv |
| Number of Records | 800 |
| Data Type | Structured assessment data |
| Target | Skill proficiency classification |
| Training and Testing Split | Stratified 80:20 split |

### Dataset Features

The dataset contains assessment-related information representing the competency areas used by the application.

These features support the classification of students into proficiency categories.

### Preprocessing

The dataset preparation and preprocessing process is handled through the project's data-processing and machine learning modules.

The prepared dataset is used to train and evaluate the classification model.

**Dataset limitation:** The dataset is synthetic. Therefore, the model's reported performance should not be interpreted as independently validated performance on real-world student assessment data.

[View Dataset](https://github.com/Teja-Varshith-111/inclusive-skill-assessment/blob/main/students_dataset.csv)

---

## 7. Machine Learning Methodology

### Random Forest Classifier

The project uses a Random Forest classifier for skill proficiency prediction.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to produce a classification result.

### Why Random Forest?

The algorithm was selected because:

- It supports multi-class classification.
- It can model nonlinear relationships.
- It combines multiple decision trees.
- It provides feature-importance information.
- It is suitable for structured tabular assessment data.

### Model Workflow

1. Load the assessment dataset.
2. Prepare the input features and target labels.
3. Apply the required preprocessing.
4. Split the dataset into training and testing subsets using stratified sampling.
5. Train the Random Forest classifier.
6. Generate predictions on the test data.
7. Evaluate classification performance.
8. Save the trained model for application use.

### Evaluation Metrics

The project documentation reports a test accuracy of **91.88%** on the synthetic dataset using the documented 80:20 stratified split.

This is a repository-reported result and should not be interpreted as a guarantee of accuracy on real student populations.

Other relevant evaluation measures include:

- Precision
- Recall
- F1-score
- Confusion Matrix

The model's performance should be assessed using the actual outputs produced by the evaluation code.

---

## 8. System Architecture

The application follows a modular architecture.

```text
             USER / EVALUATOR
                    |
                    v
            STREAMLIT INTERFACE
                    |
                    v
          SKILL ASSESSMENT INPUTS
                    |
                    v
          DATA PROCESSING MODULE
                    |
                    v
          RANDOM FOREST MODEL
                    |
                    v
          PREDICTED SKILL LEVEL
                    |
          +---------+----------+
          |                    |
          v                    v
   EXPLAINABLE AI         FAIRNESS ANALYSIS
          |                    |
          +---------+----------+
                    |
                    v
             HUMAN REVIEW
                    |
                    v
             SQLITE DATABASE
                    |
                    v
           ASSESSMENT HISTORY
```

### Major Application Modules

| Module | Responsibility |
|---|---|
| app.py | Streamlit application and user interface |
| model.py | Machine learning prediction functionality |
| dataset.py | Dataset-related operations |
| fairness.py | Group-level fairness analysis |
| database.py | Database operations and assessment history |
| skill_classifier.pkl | Serialized trained classification model |

Refer to the current repository for the complete implementation and supporting modules.

---

## 9. Application Workflow

The application follows this general workflow:

1. The user opens the deployed Streamlit application.
2. The user enters the required assessment information.
3. The application processes the assessment inputs.
4. The trained machine learning model generates a prediction.
5. The system displays the predicted proficiency category.
6. Explainability functionality provides insights into relevant features.
7. Fairness analysis helps examine group-level outcome differences.
8. An evaluator can review the prediction and make an override when appropriate.
9. Assessment information is maintained in the database for historical reference.

---

## 10. Database Design

The project uses SQLite for assessment-related data management.

The database functionality supports:

- Storing assessment records.
- Maintaining assessment history.
- Supporting human review and prediction overrides.
- Retrieving previously recorded assessment information.

The database implementation is maintained in the `database.py` module.

Database records should be treated as sensitive where they contain identifiable student information. Do not publish genuine student records, credentials or confidential assessment information in a public repository.

---

## 11. Installation and Execution

Follow these steps to run the project locally.

### Prerequisites

Make sure the following are installed:

- Python (compatible with the project dependencies).
- pip package manager.
- Git.
- A code editor such as Visual Studio Code.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Teja-Varshith-111/inclusive-skill-assessment.git
```

### Step 2: Navigate to the Project Directory

```bash
cd inclusive-skill-assessment
```

### Step 3: Create a Virtual Environment

For Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 12. Testing and Validation

The project includes test files and testing documentation.

Testing is intended to cover:

- Application functionality.
- Assessment input handling.
- Machine learning prediction behaviour.
- Database operations.
- Fairness analysis.
- Explainability functionality.
- Human review and prediction override operations.

To execute the repository's test runner, use:

```bash
python run_all_test_cases.py
```

Review the terminal output to determine the actual test results.

Testing results should be documented using the output from the latest test execution. Avoid reporting test counts or performance values that have not been verified.

---

## 13. Individual Contributions

### Akula Teja Varshith (12510396)

**Frontend Development, Integration and Deployment**

Responsibilities:

- Designed and developed the Streamlit user interface.
- Implemented the skill-assessment user workflow and input handling.
- Integrated machine learning predictions with the application.
- Integrated explainability and fairness visualizations into the dashboard.
- Managed application deployment through Streamlit Cloud.
- Prepared project presentation and demonstration materials.

### Pullannagari Sai Vishaal Mohan (12519729)

**AI/ML Development, Database and Testing**

Responsibilities:

- Prepared and preprocessed the assessment dataset.
- Developed and trained the Random Forest classification model.
- Evaluated the machine learning model using relevant performance metrics.
- Implemented feature-importance and fairness calculations.
- Managed SQLite database operations and assessment history.
- Conducted functional testing and documented test results.

### Shared Responsibilities

Both team members participated in:

- Project requirement analysis.
- System design discussions.
- Integration and verification.
- Overall project development and review.

---

## 14. Limitations

The current system has the following limitations:

1. The model is trained using synthetic assessment data.
2. Real-world generalization has not been established.
3. Fairness analysis depends on the available assessment groups and data.
4. Automated predictions may not capture every aspect of a student's abilities.
5. Human review remains important when interpreting assessment results.
6. Model performance may change when evaluated on different datasets.

---

## 15. Future Scope

Possible future improvements include:

- Training and evaluating the model using properly governed real-world datasets.
- Adding more competency areas and assessment parameters.
- Improving fairness evaluation using additional metrics.
- Introducing advanced explainability techniques.
- Supporting multiple institutions and evaluator accounts.
- Adding secure authentication and role-based access control.
- Providing detailed progress tracking and downloadable student reports.
- Conducting broader validation across different student populations.

---

## 16. Practical Applications and Societal Impact

The system can potentially support:

- Educational institutions conducting skill assessments.
- Educators analysing student competency levels.
- Training organizations evaluating learning progress.
- Academic placement and skill-development initiatives.
- Identifying areas where additional student support may be useful.

Its focus on explainability, group-level analysis and human oversight aims to encourage more transparent and responsible use of AI-assisted assessment.

These are potential applications; effectiveness in real educational settings requires further evaluation.

---

## 17. Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/Teja-Varshith-111/inclusive-skill-assessment |
| Live Application | https://inclusive-skill-assessment.streamlit.app/ |
| Dataset | [students_dataset.csv](https://github.com/Teja-Varshith-111/inclusive-skill-assessment/blob/main/students_dataset.csv) |

---

## 18. References and Resources

1. Python Documentation – https://docs.python.org/3/
2. Streamlit Documentation – https://docs.streamlit.io/
3. Scikit-learn Documentation – https://scikit-learn.org/stable/
4. Pandas Documentation – https://pandas.pydata.org/docs/
5. NumPy Documentation – https://numpy.org/doc/
6. SQLite Documentation – https://www.sqlite.org/docs.html
7. GitHub Documentation – https://docs.github.com/

---

## 19. Conclusion

The Inclusive Assessment and Skill-Evaluation Decision Support System demonstrates how machine learning can be integrated into an educational assessment application.

By combining skill classification, explainable predictions, fairness analysis, human review and database-supported assessment history, the project provides a modular framework for AI-assisted student skill evaluation.

The project emphasizes that machine learning predictions should be interpreted responsibly and supported by human judgement, particularly in educational decision-making.

---

**Project:** Inclusive Assessment and Skill-Evaluation Decision Support System  
**Source:** Smart India Hackathon 2024  
**Mapped ID:** SIH1777  
**Team:** Akula Teja Varshith and Pullannagari Sai Vishaal Mohan

