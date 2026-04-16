import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # IntelliJ-safe backend
import matplotlib.pyplot as plt

import sklearn
print("sklearn version:", sklearn.__version__)

# --------------------- PATH & DATA LOADING ---------------------

csv_path = "synthea/output/csv/"

patients = pd.read_csv(csv_path + "patients.csv")
conditions = pd.read_csv(csv_path + "conditions.csv")
observations = pd.read_csv(csv_path + "observations.csv")
encounters = pd.read_csv(csv_path + "encounters.csv")
procedures = pd.read_csv(csv_path + "procedures.csv")
medications = pd.read_csv(csv_path + "medications.csv")

print("Patients:", patients.shape)
print("Conditions:", conditions.shape)
print("Observations:", observations.shape)
print("Encounters:", encounters.shape)
print("Procedures:", procedures.shape)
print("Medications:", medications.shape)

# --------------------- MORTALITY & AGE -------------------------

patients["MORTALITY"] = patients["DEATHDATE"].notnull().astype(int)
patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"])
patients["AGE"] = (datetime.now() - patients["BIRTHDATE"]).dt.days // 365

print(patients[["Id", "BIRTHDATE", "DEATHDATE", "MORTALITY", "AGE"]].head())

# --------------------- LAST VITALS PER PATIENT -----------------

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

print("Vitals (wide):")
print(obs_wide.head())

# --------------------- MERGE BASE TABLE ------------------------

data = patients.merge(
    obs_wide, left_on="Id", right_on="PATIENT", how="left"
)

# --------------------- CONDITIONS FEATURES ---------------------

if not conditions.empty:
    cond_counts = conditions.groupby("PATIENT").size().reset_index(
        name="NUM_CONDITIONS"
    )
    cond_feat = cond_counts

    if "CODE" in conditions.columns:
        cond_unique = (
            conditions.groupby("PATIENT")["CODE"]
            .nunique()
            .reset_index(name="NUM_UNIQUE_CONDITIONS")
        )
        cond_feat = cond_feat.merge(cond_unique, on="PATIENT", how="left")

    cond_feat = cond_feat.rename(columns={"PATIENT": "Id"})
    data = data.merge(cond_feat, on="Id", how="left")

# --------------------- ENCOUNTERS FEATURES + RECENCY -----------

if not encounters.empty:
    enc_counts = encounters.groupby("PATIENT").size().reset_index(
        name="NUM_ENCOUNTERS"
    )
    enc_feat = enc_counts

    if "ENCOUNTERCLASS" in encounters.columns:
        enc_class = (
            encounters.groupby(["PATIENT", "ENCOUNTERCLASS"])
            .size()
            .unstack(fill_value=0)
            .add_prefix("ENC_")
            .reset_index()
        )
        enc_feat = enc_feat.merge(enc_class, on="PATIENT", how="left")

    # Recency: days since last encounter (timezone-safe using UTC)
    if "START" in encounters.columns:
        encounters["START"] = pd.to_datetime(
            encounters["START"],
            errors="coerce",
            utc=True  # make times tz-aware in UTC
        )

        last_enc = (
            encounters.groupby("PATIENT")["START"]
            .max()
            .reset_index(name="LAST_ENCOUNTER_DATE")
        )

        # Compute days since last encounter using UTC "now"
        last_enc["DAYS_SINCE_LAST_ENC"] = (
                pd.Timestamp.now(tz="UTC") - last_enc["LAST_ENCOUNTER_DATE"]
        ).dt.days

        enc_feat = enc_feat.merge(
            last_enc[["PATIENT", "DAYS_SINCE_LAST_ENC"]],
            on="PATIENT",
            how="left"
        )

    enc_feat = enc_feat.rename(columns={"PATIENT": "Id"})
    data = data.merge(enc_feat, on="Id", how="left")

# --------------------- PROCEDURES FEATURES ---------------------

if not procedures.empty:
    proc_counts = (
        procedures.groupby("PATIENT")
        .size()
        .reset_index(name="NUM_PROCEDURES")
    )
    proc_counts = proc_counts.rename(columns={"PATIENT": "Id"})
    data = data.merge(proc_counts, on="Id", how="left")

# --------------------- MEDICATIONS FEATURES --------------------

if not medications.empty:
    med_counts = (
        medications.groupby("PATIENT")
        .size()
        .reset_index(name="NUM_MEDICATIONS")
    )
    med_counts = med_counts.rename(columns={"PATIENT": "Id"})
    data = data.merge(med_counts, on="Id", how="left")

# --------------------- FEATURE LISTS ---------------------------

baseline_features = ["AGE", "GENDER", "RACE"] + key_vitals

extra_numeric_features = []
for col in [
    "NUM_CONDITIONS",
    "NUM_UNIQUE_CONDITIONS",
    "NUM_ENCOUNTERS",
    "NUM_PROCEDURES",
    "NUM_MEDICATIONS",
    "DAYS_SINCE_LAST_ENC",
]:
    if col in data.columns:
        extra_numeric_features.append(col)

enc_class_cols = [c for c in data.columns if c.startswith("ENC_")]
extra_numeric_features.extend(enc_class_cols)

all_features = baseline_features + extra_numeric_features

# --------------------- BUILD DATAFRAME (ALL PATIENTS) ----------

baseline_all = data[["Id"] + all_features + ["MORTALITY"]].copy()
baseline_all = baseline_all.dropna(subset=["AGE", "GENDER", "RACE"])

print("Baseline dataset shape (ALL patients):", baseline_all.shape)
print("Class counts:")
print(baseline_all["MORTALITY"].value_counts())

X = baseline_all[all_features]
y = baseline_all["MORTALITY"]

# --------------------- TRAIN / TEST SPLIT ----------------------

value_counts = y.value_counts()
if len(value_counts) >= 2 and value_counts.min() >= 2:
    stratify_labels = y
else:
    print(
        "Not enough samples in one of the classes to stratify safely. "
        "Proceeding without stratify."
    )
    stratify_labels = None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels
)

numeric_features = [col for col in all_features if col not in ["GENDER", "RACE"]]
categorical_features = [col for col in all_features if col in ["GENDER", "RACE"]]

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

# --------------------- BASE MODEL (for tuning) -----------------

base_tree = DecisionTreeClassifier(
    random_state=42
)

ada = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", ada)
])

# --------------------- GRID SEARCH TUNING ----------------------

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__learning_rate": [0.1, 0.5, 1.0],
    "classifier__estimator__max_depth": [1, 2, 3],
}

grid = GridSearchCV(
    clf,
    param_grid,
    scoring="f1",   # focus on F1 (balance precision/recall)
    cv=5,
    n_jobs=-1
)

print("Starting GridSearchCV...")
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_clf = grid.best_estimator_

# --------------------- EVALUATE BEST MODEL ---------------------

y_pred = best_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("=" * 80)
print("Extended Model (ALL pts, tuned AdaBoost, recency feature)")
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Alive", "Dead"],
    yticklabels=["Alive", "Dead"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned AdaBoost (ALL pts)")
plt.tight_layout()
plt.savefig("adaboost_confusion_matrix_tuned_recency.png", bbox_inches="tight")
plt.close()

print("Model training and evaluation complete.")
print("Confusion matrix saved as adaboost_confusion_matrix_tuned_recency.png")
