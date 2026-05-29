# Logistic regression baseline for mortality prediction on Synthea-generated EHR data.
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
)
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

from s3_utils import save_and_upload_plot, upload_file_to_s3

csv_path = "synthea/output/csv/"

patients = pd.read_csv(csv_path + "patients.csv")
conditions = pd.read_csv(csv_path + "conditions.csv")
observations = pd.read_csv(csv_path + "observations.csv")
encounters = pd.read_csv(csv_path + "encounters.csv")

print("Patients:", patients.shape)
print("Conditions:", conditions.shape)
print("Observations:", observations.shape)
print("Encounters:", encounters.shape)

# --- Mortality flag and age ---
patients["MORTALITY"] = patients["DEATHDATE"].notnull().astype(int)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365

# --- Last observation per patient for key vitals ---
key_vitals = [
    "Body mass index (BMI) [Ratio]",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate",
]

obs_filtered = observations[observations["DESCRIPTION"].isin(key_vitals)]

obs_latest = (
    obs_filtered.sort_values("DATE")
    .groupby(["PATIENT", "DESCRIPTION"])
    .last()
    .reset_index()
)

obs_wide = (
    obs_latest.pivot(index="PATIENT", columns="DESCRIPTION", values="VALUE")
    .reset_index()
)
obs_wide.columns.name = None

data = patients.merge(obs_wide, left_on="Id", right_on="PATIENT", how="left")

features = ["AGE", "GENDER", "RACE"] + key_vitals
X = data[features]
y = data["MORTALITY"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ["AGE"] + key_vitals
categorical_features = ["GENDER", "RACE"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

bucket_name = "asghar-model-output"
folder = "ml_outputs/logistic_regression"
save_and_upload_plot(plt, bucket_name, folder=folder, filename="confusion_matrix.png")

# --- ROC Curve ---
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
save_and_upload_plot(plt, bucket_name, folder=folder, filename="roc_curve.png")

# --- Metrics Summary ---
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
    f"Model: Logistic Regression\n"
    f"Accuracy:  {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall:    {rec:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
    f"AUC:       {roc_auc:.4f}\n"
)
summary_filename = "metrics_summary.txt"
with open(summary_filename, "w") as f:
    f.write(summary_text)

upload_file_to_s3(summary_filename, bucket_name, f"{folder}/{summary_filename}")

print("Model training and evaluation complete.")
