# DiabetesCare AI: Explainable Diabetes Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-blueviolet.svg?style=flat)](https://github.com/shap/shap)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org)

An end-to-end clinical machine learning web platform that screens diabetes risk from clinical measurements, transparently explains every individual prediction using game-theoretic **SHAP (SHapley Additive exPlanations)** values, and provides tailored, role-based workflows for **Patients** and **Healthcare Administrators**.

Developed for the **AI for Healthcare** curriculum.

---

## Key Features

- **Clinical Decision Support:** High-sensitivity `RandomForestClassifier` trained on the Pima Indians Diabetes Database, optimized with `GridSearchCV` 5-fold cross-validation prioritizing **Recall** to minimize false negatives.
- **Explainable AI (SHAP):**
  - **Local Attribution:** Per-patient waterfall bar chart displaying exact positive (risk-increasing) and negative (protective) feature contributions.
  - **Plain-English Translation:** Automatic linguistic generation converting mathematical attributions into clinical sentences (e.g., *"Your Glucose (168 mg/dL) increased your risk the most"*).
  - **Global Explainability:** Full-population SHAP beeswarm summary plots for clinical governance and audits.
- **Robust Zero-Handling & Imputation:** Automatically replaces physiologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with training medians (calculated strictly on the training split to eliminate data leakage).
- **Role-Based Access Control (RBAC):**
  - **Patient / User:** Register, log in, submit clinical indicators, view instant explainable risk scores, access personal prediction history, and download official clinical assessment reports as PDF.
  - **Administrator:** Seeded administrator login, population analytics dashboard, user lifecycle management (activate/deactivate/delete/reset password), global predictions log, and CSV audit export.
- **Security & Reliability:**
  - Password hashing with salted `bcrypt`.
  - Strictly parameterized SQL queries against SQLite (`db/database.py`).
  - Cached model artifacts and explainers via `@st.cache_resource`.

---

## Repository Structure

```
DiaLogix/
├── README.md                    # System documentation & setup guide
├── requirements.txt             # Dependency list
├── seed_admin.py                # Administrator account seeding script
├── app.py                       # Portal entry point (Login/Register & Routing)
├── ui_common.py                 # Shared clinical UI tokens, theme, and PDF generator
│
├── data/
│   ├── diabetes.csv             # Raw Pima Indians Diabetes dataset
│   └── download_data.py         # Automated dataset verification script
│
├── notebooks/
│   └── 01_eda_and_training.ipynb # Exploratory data analysis & experiments
│
├── ml/
│   ├── preprocess.py            # Zero-handling & training-median imputation
│   ├── train.py                 # Pipeline: GridSearchCV, evaluation, SHAP artifacts
│   ├── predict.py               # Validation, inference, risk bands, recommendations
│   └── explain.py               # Local & global SHAP attributions and plotting
│
├── artifacts/
│   ├── model.joblib             # Trained Random Forest model
│   ├── imputer.joblib           # Pickled training medians
│   ├── medians.json             # JSON training medians
│   ├── explainer.joblib         # SHAP TreeExplainer
│   ├── metrics.json             # Test-set Recall, Accuracy, ROC-AUC, Confusion Matrix
│   └── global_shap.png          # High-resolution global beeswarm plot
│
├── db/
│   ├── schema.sql               # SQLite schema (users & predictions tables)
│   └── database.py              # Parameterized CRUD operations & analytics queries
│
├── auth/
│   └── auth.py                  # Bcrypt hashing, sessions, and role guards
│
└── pages/
    ├── 1_Predict.py             # User: Input form + result + SHAP + PDF report
    ├── 2_My_History.py          # User: Past predictions log & re-inspection
    ├── 3_About_Model.py         # User: Model specs, metrics, global SHAP, limits
    ├── 4_Admin_Dashboard.py     # Admin: Analytics, timelines, risk distribution
    ├── 5_Admin_Users.py         # Admin: Account directory, activation, password reset
    └── 6_Admin_Predictions.py   # Admin: Full database audit & CSV export
```

---

## Quickstart Guide

### 1. Installation

Clone the repository and install required packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Model & Generate Artifacts

Run the training script to perform hyperparameter tuning, compute training medians, generate test-set metrics, and build SHAP explainers:

```bash
python ml/train.py
```

All trained artifacts will be saved into the `artifacts/` folder:
- `artifacts/model.joblib`
- `artifacts/medians.json`
- `artifacts/metrics.json`
- `artifacts/explainer.joblib`
- `artifacts/global_shap.png`

### 3. Seed the Administrator Account

Seed the initial administrator credentials into the SQLite database:

```bash
python seed_admin.py
```

*Default Admin Credentials:*
- **Email:** `admin@diabetescare.ai`
- **Password:** `Admin@123`

*(Custom credentials can be supplied via environment variables `ADMIN_EMAIL` and `ADMIN_PASSWORD` or arguments `--email` and `--password`)*.

### 4. Launch the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Clinical Validation & Input Ranges

The prediction form enforces physiological validation ranges:

| Feature | Range | Clinical Interpretation |
|---|---|---|
| **Pregnancies** | 0 – 20 | Number of times pregnant (integer) |
| **Glucose** | 40 – 300 mg/dL | 2-hour oral glucose tolerance test |
| **Blood Pressure** | 30 – 200 mm Hg | Diastolic blood pressure |
| **Skin Thickness** | 0 – 100 mm | Triceps skin fold thickness (*0 = imputed*) |
| **Insulin** | 0 – 900 μU/mL | 2-hour serum insulin (*0 = imputed*) |
| **BMI** | 10.0 – 70.0 | Body Mass Index ($kg/m^2$) |
| **Diabetes Pedigree**| 0.05 – 2.50 | Genetic family history score |
| **Age** | 1 – 120 | Patient age in completed years |

### Risk Tiers

- **Low Risk:** Predicted probability &lt; 30%
- **Moderate Risk:** Predicted probability 30% – 60%
- **High Risk:** Predicted probability &gt; 60%

---

## Security Architecture

- **Bcrypt Hashing:** Passwords are never stored in plaintext; all authentication is verified via salted bcrypt hashes.
- **SQL Injection Prevention:** 100% of SQLite database transactions execute through parameterized queries.
- **Route & Page Guards:** Every individual page in `pages/` runs `require_auth(['admin'])` or `require_auth(['user', 'admin'])`. Unauthenticated requests are halted with `st.stop()` and redirected to the login portal.
- **Session Isolation:** Users can only view their own prediction records. Admins can audit all records.

---

## Medical & Educational Disclaimer

> **IMPORTANT:** This application is built strictly for **educational and research demonstration purposes** in an AI for Healthcare course. It does **not** constitute medical advice, diagnosis, or clinical recommendation. All health and treatment decisions must be made in consultation with a qualified medical professional.
