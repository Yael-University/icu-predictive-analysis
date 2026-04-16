# Logistic Regression to predict patient mortality based on Synthea-generated EHR data (LOCAL VERSION)

import os
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------- CONFIG ------------------------------------------------------------------

# Path to synthea folder (adjust this path if needed)
csv_path = "synthea/output/csv/"

# Local output directory for plots and metrics
output_dir = "ml_outputs/logistic_regression"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------- LOAD DATA ---------------------------------------------------------------

# Load main CSVs from Synthea
patients = pd.read_csv(os.path.join(csv_path, "patients.csv"))
conditions = pd.read_csv(os.path.join(csv_path, "conditions.csv"))
observations = pd.read_csv(os.path.join(csv_path, "observations.csv"))
encounters = pd.read_csv(os.path.join(csv_path, "encounters.csv"))

print("Observation columns:", observations.columns.tolist())
print(observations.head())

print("Patients:", patients.shape)
print("Conditions:", conditions.shape)
print("Observations:", observations.shape)
print("Encounters:", encounters.shape)

# ----------------------------- MORTALITY & AGE ---------------------------------------------------------

patients["MORTALITY"] = patients["DEATHDATE"].notnull().astype(int)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365

print(patients[["Id", "BIRTHDATE", "DEATHDATE", "MORTALITY", "AGE"]].head())

# ----------------------------- LAST VITALS PER PATIENT -------------------------------------------------

key_vitals = [
    "Body mass index (BMI) [Ratio]",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
]

# Keep only the above vital rows
obs_filtered = observations[observations["DESCRIPTION"].isin(key_vitals)]

# Sort by DATE and keep the last observation per patient per vital
obs_latest = (
    obs_filtered.sort_values("DATE")
    .groupby(["PATIENT", "DESCRIPTION"])
    .last()
    .reset_index()
)

# Pivot to wide format (one row per patient, vitals as columns)
obs_wide = (
    obs_latest.pivot(index="PATIENT", columns="DESCRIPTION", values="VALUE")
    .reset_index()
)

# Remove multi-index column naming
obs_wide.columns.name = None

print("Vitals (wide):")
print(obs_wide.head())

# ----------------------------- MERGE WITH PATIENTS -----------------------------------------------------

data = patients.merge(obs_wide, left_on="Id", right_on="PATIENT", how="left")

# Features and target
features = ["AGE", "GENDER", "RACE"] + key_vitals
X = data[features]
y = data["MORTALITY"]

print(X.head())
print(y.head())

# ----------------------------- TRAIN / TEST SPLIT ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
numeric_features = ["AGE"] + key_vitals
categorical_features = ["GENDER", "RACE"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ----------------------------- MODEL PIPELINE ----------------------------------------------------------

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

# Train
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ----------------------------- CONFUSION MATRIX --------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Alive", "Dead"],
    yticklabels=["Alive", "Dead"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# ----------------------------- ROC CURVE ---------------------------------------------------------------

y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")

roc_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_path, bbox_inches="tight")
plt.close()
print(f"ROC curve saved to: {roc_path}")

# ----------------------------- METRICS SUMMARY ---------------------------------------------------------

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Logistic Regression Metrics Summary ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {roc_auc:.4f}")
print("==========================================\n")

summary_text = (
    "Model: Logistic Regression\n"
    f"Accuracy:  {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall:    {rec:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
    f"AUC:       {roc_auc:.4f}\n"
)

summary_filename = os.path.join(output_dir, "metrics_summary.txt")
with open(summary_filename, "w") as f:
    f.write(summary_text)

print(f"Metrics summary saved to: {summary_filename}")
print("Model training and evaluation complete (local version).")
