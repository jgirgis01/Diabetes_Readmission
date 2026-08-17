import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. File paths
raw_csv_path = "examplepath.csv"
tableau_csv_path = "examplepath2.csv"

# 2. In-memory SQL database for data ingestion & cleaning
conn = sqlite3.connect(":memory:")
raw_df = pd.read_csv(raw_csv_path)
raw_df.to_sql("raw_diabetic_data", conn, index=False, if_exists="replace")

sql_query = """
SELECT
    encounter_id,
    patient_nbr,
    race,
    gender,
    age AS Age_Bracket,
    weight,
    admission_type_id,
    discharge_disposition_id,
    admission_source_id,
    time_in_hospital AS Length_of_Stay_Days,
    payer_code,
    medical_specialty,
    num_lab_procedures AS Num_Lab_Procedures,
    num_procedures AS Num_Procedures,
    num_medications AS Total_Medications,
    number_outpatient AS Prior_Outpatient_Visits,
    number_emergency AS Prior_Emergency_Visits,
    number_inpatient AS Prior_Inpatient_Visits,
    diag_1,
    diag_2,
    diag_3,
    number_diagnoses AS Num_Diagnoses,
    max_glu_serum,
    A1Cresult,
    metformin,
    repaglinide,
    nateglinide,
    chlorpropamide,
    glimepiride,
    acetohexamide,
    glipizide,
    glyburide,
    tolbutamide,
    pioglitazone,
    rosiglitazone,
    acarbose,
    miglitol,
    troglitazone,
    tolazamide,
    examide,
    citoglipton,
    insulin,
    `glyburide-metformin`,
    `glipizide-metformin`,
    `glimepiride-pioglitazone`,
    `metformin-rosiglitazone`,
    `metformin-pioglitazone`,
    change AS med_change,
    diabetesMed,
    readmitted,

    -- Target Flags
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS Is_Readmitted_30d,
    CASE WHEN readmitted != 'NO' THEN 1 ELSE 0 END AS Is_Readmitted_Any,

    -- ICD-9 Primary Diagnosis Categorization
    CASE 
        WHEN (diag_1 >= '390' AND diag_1 <= '459') OR diag_1 = '785' THEN 'Circulatory'
        WHEN diag_1 LIKE '250%' THEN 'Diabetes'
        WHEN (diag_1 >= '460' AND diag_1 <= '519') OR diag_1 = '786' THEN 'Respiratory'
        WHEN (diag_1 >= '520' AND diag_1 <= '579') OR diag_1 = '787' THEN 'Digestive'
        WHEN (diag_1 >= '580' AND diag_1 <= '629') OR diag_1 = '788' THEN 'Genitourinary'
        WHEN diag_1 >= '140' AND diag_1 <= '239' THEN 'Neoplasms (Cancer)'
        WHEN diag_1 >= '710' AND diag_1 <= '739' THEN 'Musculoskeletal'
        WHEN diag_1 >= '800' AND diag_1 <= '999' THEN 'Injury / Poisoning'
        ELSE 'Other Diagnoses'
    END AS Primary_Diagnosis_Group,

    -- Discharge Destination Clinical Mapping
    CASE discharge_disposition_id
        WHEN 1  THEN 'Discharged to home'
        WHEN 2  THEN 'Transferred to short-term hospital'
        WHEN 3  THEN 'Discharged to SNF'
        WHEN 4  THEN 'Discharged to ICF'
        WHEN 5  THEN 'Transferred to other inpatient facility'
        WHEN 6  THEN 'Home with home health service'
        WHEN 7  THEN 'Left AMA'
        WHEN 8  THEN 'Home IV provider'
        WHEN 11 THEN 'Expired'
        WHEN 13 THEN 'Hospice / facility'
        WHEN 14 THEN 'Hospice / home'
        WHEN 18 THEN 'Transferred within facility'
        WHEN 22 THEN 'Discharged to rehab'
        ELSE 'Other / Not Mapped'
    END AS Discharge_Disposition_Desc,

    -- Admission Type Mapping
    CASE admission_type_id
        WHEN 1 THEN 'Emergency'
        WHEN 2 THEN 'Urgent'
        WHEN 3 THEN 'Elective'
        WHEN 4 THEN 'Newborn'
        WHEN 7 THEN 'Trauma Center'
        ELSE 'Other / Unknown'
    END AS Admission_Type_Desc

FROM raw_diabetic_data

-- Exclude in-hospital mortality and hospice discharges
WHERE discharge_disposition_id NOT IN (11, 13, 14, 19, 20, 21);
"""

df = pd.read_sql_query(sql_query, conn)

# 3. Train Logistic Regression Model (Statistical ML Foundation)
feature_cols = [
    "Prior_Inpatient_Visits",
    "Prior_Emergency_Visits",
    "Prior_Outpatient_Visits",
    "Length_of_Stay_Days",
    "Total_Medications",
    "Num_Lab_Procedures",
    "Num_Procedures",
    "Num_Diagnoses",
]

cat_dummies = pd.get_dummies(
    df[["Discharge_Disposition_Desc", "Primary_Diagnosis_Group", "A1Cresult"]],
    drop_first=True,
    dtype=float,
)

X = pd.concat([df[feature_cols].fillna(0), cat_dummies], axis=1)
y = df["Is_Readmitted_30d"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_model.fit(X_scaled, y)

# Assign statistical Logistic Regression probabilities
df["Predicted_Readmit_Probability"] = lr_model.predict_proba(X_scaled)[:, 1]

# ---------------------------------------------------------
# EXTRACT ODDS RATIOS FOR TABLEAU
# ---------------------------------------------------------
odds_ratios_path = r"C:\Users\jacki\Documents\Job applications\Example Dashboard\Diabetes\feature_odds_ratios.csv"

odds_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient_Beta": lr_model.coef_[0],
    "Odds_Ratio": np.exp(lr_model.coef_[0])
}).sort_values(by="Odds_Ratio", ascending=False)

# Add a clean classification for Tableau coloring
odds_df["Impact_Type"] = np.where(
    odds_df["Odds_Ratio"] > 1.05, "Increases Risk",
    np.where(odds_df["Odds_Ratio"] < 0.95, "Protective / Lowers Risk", "Neutral")
)

# Export to CSV
odds_df.to_csv(odds_ratios_path, index=False)
print(f"Exported Odds Ratios table to: {odds_ratios_path}")


# 4. Point-Scoring Multipliers (Clinical Heuristic Risk Score)
df["pts_inpatient"] = np.where(df["Prior_Inpatient_Visits"] >= 2, 4, np.where(df["Prior_Inpatient_Visits"] == 1, 2, 0))
df["pts_emergency"] = np.where(df["Prior_Emergency_Visits"] >= 2, 2, np.where(df["Prior_Emergency_Visits"] == 1, 1, 0))
df["pts_los"] = np.where(df["Length_of_Stay_Days"] >= 7, 3, np.where(df["Length_of_Stay_Days"] >= 4, 1, 0))
df["pts_meds"] = np.where(df["Total_Medications"] >= 16, 2, 0)
df["pts_diagnoses"] = np.where(df["Num_Diagnoses"] >= 8, 2, 0)
df["pts_discharge"] = np.where(df["Discharge_Disposition_Desc"].isin(["Discharged to SNF", "Discharged to rehab", "Home with home health service"]), 2, 0)
df["pts_a1c"] = np.where(df["A1Cresult"] == ">8", 2, 0)

df["Clinical_Risk_Score"] = (
    df["pts_inpatient"] + df["pts_emergency"] + df["pts_los"] +
    df["pts_meds"] + df["pts_diagnoses"] + df["pts_discharge"] + df["pts_a1c"]
)

# 5. Risk Tiering Assignment
df["Clinical_Risk_Tier"] = np.select(
    [
        df["Clinical_Risk_Score"] <= 3,
        (df["Clinical_Risk_Score"] > 3) & (df["Clinical_Risk_Score"] <= 7),
        df["Clinical_Risk_Score"] > 7
    ],
    ["Low Risk", "Moderate Risk", "High Risk Priority"],
    default="Moderate Risk"
)

# 7. Export Final Dataset to CSV for Tableau
df.to_csv(tableau_csv_path, index=False)

print("--- Pipeline Successfully Executed ---")
print(f"Logistic Regression coefficients computed and model fitted.")
print(f"Point scoring multipliers and risk tiers applied.")
print(f"CSV Export saved to: {tableau_csv_path}")