import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import shap
import matplotlib.pyplot as plt
import seaborn as sns

print("SCRIPT STARTED")

DATA_DIR = "/Users/sarahmtorres/Downloads/MIMIC"

ADMISSIONS_CSV = os.path.join(DATA_DIR, "admissions.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
DIAGNOSES_CSV = os.path.join(DATA_DIR, "diagnoses_icd.csv")
PROCEDURES_CSV = os.path.join(DATA_DIR, "procedures_icd.csv")

RANDOM_STATE = 42

RISK_LOW = 0.30
RISK_HIGH = 0.70


def risk_label(prob):
    if prob >= RISK_HIGH:
        return "HIGH RISK"
    elif prob >= RISK_LOW:
        return "MODERATE RISK"
    return "LOW RISK"


def main():

    admissions = pd.read_csv(ADMISSIONS_CSV, parse_dates=["admittime"])
    patients = pd.read_csv(PATIENTS_CSV)
    dx = pd.read_csv(DIAGNOSES_CSV)
    px = pd.read_csv(PROCEDURES_CSV)

    admissions["MORTALITY"] = admissions["hospital_expire_flag"]
    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year

    base = admissions.merge(patients, on="subject_id", how="left")

    base["AGE"] = base["anchor_age"]

    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")

    feat = (
        base.set_index("hadm_id")
        .join(num_conditions, how="left")
        .join(num_procedures, how="left")
        .fillna(0)
        .reset_index()
    )

    feat["GENDER"] = feat["gender"]
    feat["RACE"] = feat["race"]

    numeric_features = ["AGE", "NUM_CONDITIONS", "NUM_PROCEDURES"]
    categorical_features = ["GENDER", "RACE"]

    target = "MORTALITY"

    data = feat[numeric_features + categorical_features + [target]].dropna()

    X = data[numeric_features + categorical_features]
    y = data[target].astype(int)

    print("Dataset:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    print("Training Random Forest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # High-risk Flagging
    probs = clf.predict_proba(X_test)[:, 1]

    print("\n--- Sample Risk Predictions ---")
    for i in range(5):
        prob = probs[i]
        print(f"Patient {i}: {prob:.2%} → {risk_label(prob)}")

    # SHAP Explainability
    print("\nGenerating SHAP explanations...")

    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["classifier"]

    X_sample = X_test.iloc[:200]
    X_processed = preprocessor.transform(X_sample)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values[:, :, 1]

    # Global importance
    shap.summary_plot(shap_values_class1, X_processed, feature_names=feature_names)

    # Explain one patient
    idx = 0
    print("\n--- Explanation for Patient 0 ---")

    shap_vals = shap_values_class1[idx]
    top_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]

    for i in top_idx:
       print(f"{feature_names[i]}: {shap_vals[i]:+.4f}")


if __name__ == "__main__":
    main()