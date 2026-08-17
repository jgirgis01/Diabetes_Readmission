# Clinical Analytics & 30-Day Diabetes Readmission Pipeline

An end-to-end healthcare data science and clinical analytics pipeline that predicts 30-day hospital readmissions among diabetic patients. 

This project integrates in-memory SQL extraction, machine learning classification, and clinical risk scoring to surface actionable patient cohorts for care coordinators and clinical teams.

**Interactive Tableau Dashboard:** [View Live Dashboard on Tableau Public](https://public.tableau.com/app/profile/jackie.girgis/viz/JackieGirgis-DiabetesReadmissionDashboard/Dashboard1?publish=yes)

---

## Dataset & Clinical Source

This project utilizes the **Diabetes 130-US Hospitals for Years 1999–2008 Dataset** from the UCI Machine Learning Repository.

* **Source:** [UCI Machine Learning Repository – Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
* **Dataset Scope:** 10 years of clinical care data representing 101,766 inpatient encounters across 130 US hospitals.
* **Key Attributes:** Demographics, admission/discharge clinical dispositions, length of stay, laboratory diagnostics (HbA1c, glucose), 23 diabetes medications, ICD-9 primary/secondary diagnoses, and 30-day readmission outcomes.

---

## Overview & Clinical Objectives

30-day readmissions represent a critical quality metric and financial penalty under hospital value-based purchasing programs. This pipeline solves two core operational challenges:

* **Statistical Risk Prediction:** Quantifies 30-day readmission risk via balanced Logistic Regression and extracts feature odds ratios ($\text{OR} = e^{\beta}$).
* **Rule-Based Clinical Triage:** Constructs an interpretable, point-based clinical risk index (Low, Moderate, High Priority) aligned with bedside decision rules.

---

## Pipeline Architecture

```text
  [ Raw Clinical Dataset (CSV) ]
                 │
                 ▼
  [ In-Memory SQLite Layer ]
    • ICD-9 primary diagnostic group mapping
    • Discharge & admission categorization
    • Exclusion criteria (mortality & hospice)
                 │
                 ▼
  [ Python Transformation & ML Engine ]
    ├─► Preprocessing: One-hot encoding & StandardScaler
    ├─► Logistic Regression (class_weight='balanced')
    ├─► Odds Ratio Calculation: exp(Beta)
    └─► Heuristic Point-Scoring & Risk Stratification
                 │
                 ▼
  [ Export Layer & Interactive BI ]
    ├─► feature_odds_ratios.csv
    └─► tableau_clinical_analytics.csv ──► Tableau Public Dashboard
```

---

## Methodology & Logic

### 1. In-Memory SQL Cleaning & Clinical Mappings
Data ingestion and cleaning are executed in an embedded SQLite engine (`:memory:`) to apply clinical exclusions and standardize medical encodings:
* **Exclusion Criteria:** Patients discharged to hospice or expired during hospitalization (`discharge_disposition_id IN (11, 13, 14, 19, 20, 21)`) are excluded from readmission risk pools.
* **ICD-9 Grouping:** Primary diagnosis codes (`diag_1`) are categorized into standard clinical clusters (Circulatory, Respiratory, Diabetes, Digestive, Genitourinary, Musculoskeletal, Neoplasms, Injury/Poisoning).
* **Discharge Destination:** Mapped to standardized destination descriptions (Home, SNF, Rehab, Home Health).

### 2. Statistical Modeling & Odds Ratio Analysis
* Features include prior utilization (`inpatient`, `emergency`, `outpatient`), `length_of_stay`, medication counts, laboratory volume, diagnostic burden, and categorical clinical features.
* A balanced `LogisticRegression` model estimates patient-level predicted probabilities (`Predicted_Readmit_Probability`).
* Feature impact is quantified via exponentiated coefficients ($\text{Odds Ratio} = e^\beta$), categorizing features into *Risk Increasing* ($\text{OR} > 1.05$), *Protective* ($\text{OR} < 0.95$), or *Neutral*.

### 3. Clinical Heuristic Risk Score & Tiering
To complement probabilistic outputs with transparent bedside rules, a cumulative point score is calculated:

| Clinical Feature | Threshold | Score Assigned |
| :--- | :--- | :--- |
| **Prior Inpatient Visits** | $\ge 2$ visits / $1$ visit | $+4$ pts / $+2$ pts |
| **Prior Emergency Visits** | $\ge 2$ visits / $1$ visit | $+2$ pts / $+1$ pt |
| **Length of Stay** | $\ge 7$ days / $\ge 4$ days | $+3$ pts / $+1$ pt |
| **Medication Burden** | $\ge 16$ total medications | $+2$ pts |
| **Diagnostic Complexity** | $\ge 8$ diagnoses | $+2$ pts |
| **High-Risk Discharge** | Discharged to SNF, Rehab, or Home Health | $+2$ pts |
| **Glycemic Control** | $\text{HbA1c} > 8\%$ | $+2$ pts |

* **Risk Tiers:** $\le 3$ Points (Low Risk) | $4 - 7$ Points (Moderate Risk) | $> 7$ Points (High Risk Priority).

---

## Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Data Ingestion & SQL:** `sqlite3`, `pandas`
* **Machine Learning & Preprocessing:** `scikit-learn` (`LogisticRegression`, `StandardScaler`), `numpy`
* **Business Intelligence:** Tableau Public

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn
   ```

3. **Run the pipeline:**
   Place the raw dataset `diabetic_data.csv` in the root directory and run:
   ```bash
   Workflow.py
   ```

---

## Deliverables & Dashboard

* **`tableau_clinical_analytics.csv`**: Master enriched dataset containing demographic data, utilization flags, predicted probabilities, and clinical risk tiers.
* **`feature_odds_ratios.csv`**: Statistical odds ratios for all model covariates.
* **[Interactive Dashboard](https://public.tableau.com/app/profile/jackie.girgis/viz/JackieGirgis-DiabetesReadmissionDashboard/Dashboard1?publish=yes)**: Live visual interface for clinical cohort drill-down, risk tier distribution, and feature impact analysis.
