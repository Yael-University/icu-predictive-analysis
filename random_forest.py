import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom S3 helper
from s3_utils import save_and_upload_plot

# Path to synthea folder
csv_path = "synthea/output/csv/"

# Load main CSVs
patients = pd.read_csv(csv_path + "patients.csv")
conditions = pd.read_csv(csv_path + "conditions.csv")
observations = pd.read_csv(csv_path + "observations.csv")
encounters = pd.read_csv(csv_path + "encounters.csv")

# --- Mortality flag & age calculation ---
patients["MORTALITY"] = patients["DEATHDATE"].notnull().astype(int)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365

# --- Filter and reshape vitals ---
key_vitals = [
    "Body mass index (BMI) [Ratio]",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart rate"
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

# --- Merge with patients ---
data = patients.merge(obs_wide, left_on="Id", right_on="PATIENT", how="left")

# --- Features and target ---
features = ["AGE", "GENDER", "RACE"] + key_vitals
X = data[features]
y = data["MORTALITY"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Preprocessing ---
numeric_features = ["AGE"] + key_vitals
categorical_features = ["GENDER", "RACE"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# --- Random Forest pipeline ---
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    ))
])

# --- Train model ---
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

bucket_name = "asghar-model-output"
folder = "ml_outputs/random_forest"
save_and_upload_plot(plt, bucket_name, folder=folder, filename="confusion_matrix.png")

# --- ROC Curve ---
y_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
save_and_upload_plot(plt, bucket_name, folder=folder, filename="roc_curve.png")

# --- Feature Importance ---
model = clf.named_steps["classifier"]
feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
importances = model.feature_importances_

import numpy as np
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importance - Random Forest")
save_and_upload_plot(plt, bucket_name, folder=folder, filename="feature_importance.png")

# --- Metrics Summary (printed and saved) --------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Random Forest Metrics Summary ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {roc_auc:.4f}")
print("=====================================\n")

# Save summary as a small text file and upload to S3
summary_text = (
    f"Model: Random Forest\n"
    f"Accuracy:  {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall:    {rec:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
    f"AUC:       {roc_auc:.4f}\n"
)
summary_filename = "metrics_summary.txt"
with open(summary_filename, "w") as f:
    f.write(summary_text)

from s3_utils import upload_file_to_s3
upload_file_to_s3(summary_filename, bucket_name, f"{folder}/{summary_filename}")

print("Random Forest training and evaluation complete.")
