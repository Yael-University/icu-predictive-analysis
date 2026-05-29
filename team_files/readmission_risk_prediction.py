import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "MIMIC"

ADMISSIONS_CSV = os.path.join(DATA_DIR, "admissions.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
DIAGNOSES_CSV = os.path.join(DATA_DIR, "diagnoses_icd.csv")
PROCEDURES_CSV = os.path.join(DATA_DIR, "procedures_icd.csv")

# Optional (only used if present)
PRESCRIPTIONS_CSV = os.path.join(DATA_DIR, "prescriptions.csv")

RANDOM_STATE = 42

# Readmission window (in days)
READMISSION_DAYS = 30


def must_exist(path: str) -> None:
    """Validate that a required file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_read_csv(path: str, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
    """Safely read a CSV file with specified options."""
    return pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )


def main() -> None:
    # 0) Validate required files
    must_exist(ADMISSIONS_CSV)
    must_exist(PATIENTS_CSV)
    must_exist(DIAGNOSES_CSV)
    must_exist(PROCEDURES_CSV)

    # 1) Load core tables
    admissions = safe_read_csv(
        ADMISSIONS_CSV,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "edregtime", "edouttime"],
        dtype={"subject_id": "int32", "hadm_id": "int32"},
    )

    patients = safe_read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        dtype={"subject_id": "int32", "anchor_age": "float32", "anchor_year": "int32"},
    )

    dx = safe_read_csv(
        DIAGNOSES_CSV,
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "int32", "icd_code": "string"},
    )

    px = safe_read_csv(
        PROCEDURES_CSV,
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "int32", "icd_code": "string"},
    )

    # Optional meds
    has_rx = os.path.exists(PRESCRIPTIONS_CSV)
    if has_rx:
        rx = safe_read_csv(
            PRESCRIPTIONS_CSV,
            usecols=["hadm_id", "drug"],
            dtype={"hadm_id": "int32", "drug": "string"},
        )
    else:
        rx = None
        print("Note: prescriptions.csv not found. Skipping medication features.")

    print("Loaded shapes:")
    print("  admissions:", admissions.shape)
    print("  patients:", patients.shape)
    print("  diagnoses:", dx.shape)
    print("  procedures:", px.shape)
    if rx is not None:
        print("  prescriptions:", rx.shape)

    # 2) Identify Readmissions
    # Remove rows where dischtime is missing
    admissions = admissions.dropna(subset=["dischtime", "admittime"])
    
    # Exclude patients who died (can't be readmitted)
    admissions = admissions[admissions["hospital_expire_flag"] == 0]
    
    # Sort by patient and discharge time
    admissions = admissions.sort_values(["subject_id", "dischtime"]).reset_index(drop=True)
    
    # Calculate LOS
    admissions["LOS_DAYS"] = (
        (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")
    
    # For each admission, get the next admission time for the same patient
    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    
    # Calculate days until next admission
    admissions["days_to_readmission"] = (
        (admissions["next_admittime"] - admissions["dischtime"]).dt.total_seconds() / 86400
    )
    
    # Create binary readmission flag
    admissions["READMITTED_30D"] = (
        (admissions["days_to_readmission"] <= READMISSION_DAYS) & 
        (admissions["days_to_readmission"] > 0)
    ).astype(int)
    
    # Remove the last admission for each patient (no follow-up data)
    admissions = admissions[admissions["next_admittime"].notna()].copy()
    
    print(f"\n{READMISSION_DAYS}-Day Readmission Statistics:")
    print(f"Total admissions with follow-up: {len(admissions)}")
    print(f"Readmissions: {admissions['READMITTED_30D'].sum()}")
    print(f"Readmission rate: {admissions['READMITTED_30D'].mean():.2%}")

    # ----------------------------
    # 3) Base features
    # ----------------------------
    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year.astype("int32")
    
    base = admissions.merge(patients, on="subject_id", how="left")

    # Age at admission
    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")

    # ED length (hours)
    base["ED_LOS_HOURS"] = ((base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600).astype("float32")

    # Flags
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["EMERGENCY_ADMISSION"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["ELECTIVE_ADMISSION"] = (base["admission_type"] == "ELECTIVE").astype("int8")

    # Demographics
    base["GENDER"] = base["gender"]
    base["RACE"] = base["race"]
    base["LANGUAGE"] = base["language"]
    
    # Discharge location is a strong predictor of readmission
    base["DISCHARGE_LOCATION"] = base["discharge_location"]

    # ----------------------------
    # 4) Previous admission history
    # ----------------------------
    # Count number of previous admissions for each patient
    base["admission_number"] = base.groupby("subject_id").cumcount() + 1
    base["PREVIOUS_ADMISSIONS"] = base["admission_number"] - 1
    
    # Calculate time since last admission (for patients with previous admissions)
    base["prev_dischtime"] = base.groupby("subject_id")["dischtime"].shift(1)
    base["DAYS_SINCE_LAST_ADMISSION"] = (
        (base["admittime"] - base["prev_dischtime"]).dt.total_seconds() / 86400
    ).astype("float32")
    base["DAYS_SINCE_LAST_ADMISSION"] = base["DAYS_SINCE_LAST_ADMISSION"].fillna(0)

    # ----------------------------
    # 5) Feature engineering from dx/px/rx
    # ----------------------------
    # Diagnoses counts
    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_unique_conditions = dx.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_CONDITIONS")

    # Procedures counts
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    num_unique_procedures = px.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_PROCEDURES")

    # ICD group features
    dx["ICD_GROUP"] = dx["icd_code"].astype("string").str[0].fillna("?")
    top_groups = dx["ICD_GROUP"].value_counts().nlargest(15).index.tolist()
    dx_top = (
        dx[dx["ICD_GROUP"].isin(top_groups)]
        .groupby(["hadm_id", "ICD_GROUP"])
        .size()
        .unstack(fill_value=0)
    )
    dx_top.columns = [f"ICD_{c}_COUNT" for c in dx_top.columns]

    # Optional meds
    if rx is not None:
        num_meds = rx.groupby("hadm_id").size().rename("NUM_MEDICATIONS")
        num_unique_meds = rx.groupby("hadm_id")["drug"].nunique().rename("NUM_UNIQUE_MEDICATIONS")
    else:
        num_meds = None
        num_unique_meds = None

    # 6) Merge into one wide table
    feat = (
        base.set_index("hadm_id")
        .join(num_conditions, how="left")
        .join(num_unique_conditions, how="left")
        .join(num_procedures, how="left")
        .join(num_unique_procedures, how="left")
        .join(dx_top, how="left")
    )

    if num_meds is not None:
        feat = feat.join(num_meds, how="left").join(num_unique_meds, how="left")

    feat = feat.reset_index().rename(columns={"hadm_id": "Id"})

    # Fill missing counts with 0
    count_cols = [
        "NUM_CONDITIONS",
        "NUM_UNIQUE_CONDITIONS",
        "NUM_PROCEDURES",
        "NUM_UNIQUE_PROCEDURES",
    ]
    if rx is not None:
        count_cols += ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]

    for c in count_cols:
        feat[c] = feat[c].fillna(0).astype("int32")

    # ICD_* columns
    icd_cols = [c for c in feat.columns if c.startswith("ICD_") and c.endswith("_COUNT")]
    for c in icd_cols:
        feat[c] = feat[c].fillna(0).astype("int32")

    # ----------------------------
    # 7) Prepare features and target
    # ----------------------------
    numeric_features = [
        "AGE",
        "HAS_ED_VISIT",
        "ED_LOS_HOURS",
        "EMERGENCY_ADMISSION",
        "ELECTIVE_ADMISSION",
        "LOS_DAYS",
        "PREVIOUS_ADMISSIONS",
        "DAYS_SINCE_LAST_ADMISSION",
        "NUM_CONDITIONS",
        "NUM_UNIQUE_CONDITIONS",
        "NUM_PROCEDURES",
        "NUM_UNIQUE_PROCEDURES",
    ] + icd_cols

    if rx is not None:
        numeric_features += ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]

    categorical_features = [
        "GENDER",
        "RACE",
        "LANGUAGE",
        "admission_type",
        "admission_location",
        "DISCHARGE_LOCATION",
        "insurance",
        "marital_status",
    ]

    target = "READMITTED_30D"

    # Drop rows missing critical features
    data = feat[numeric_features + categorical_features + [target]].dropna(
        subset=["AGE", "GENDER", "RACE", "LOS_DAYS", target]
    )

    X = data[numeric_features + categorical_features]
    y = data[target].astype(int)

    print("\nDataset ready:")
    print("  X:", X.shape, "y:", y.shape)
    print(f"  Readmission rate: {y.mean():.2%}")
    print(f"  Class distribution:\n{y.value_counts()}")

    # ----------------------------
    # 8) Preprocess + Model
    # ----------------------------
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Gradient Boosting often performs well for readmission prediction
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # Train/test split
    value_counts = y.value_counts()
    stratify_labels = y if len(value_counts) >= 2 and value_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    # ----------------------------
    # 9) Train and Evaluate
    # ----------------------------
    print(f"\nTraining {READMISSION_DAYS}-day readmission risk model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 80)
    print(f"{READMISSION_DAYS}-Day Readmission Risk Prediction Results")
    print("=" * 80)
    print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # ----------------------------
    # 10) Visualizations
    # ----------------------------
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Readmitted", "Readmitted"],
        yticklabels=["Not Readmitted", "Readmitted"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {READMISSION_DAYS}-Day Readmission Risk")
    plt.tight_layout()
    plt.savefig("readmission_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("\nConfusion matrix saved as readmission_confusion_matrix.png")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{READMISSION_DAYS}-Day Readmission ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("readmission_roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved as readmission_roc_curve.png (AUC: {roc_auc:.3f})")

    # Risk Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='Not Readmitted', color='green')
    plt.hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='Readmitted', color='red')
    plt.xlabel('Predicted Readmission Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Readmission Probabilities')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("readmission_risk_distribution.png", bbox_inches="tight")
    plt.close()
    print("Risk distribution saved as readmission_risk_distribution.png")

    # ----------------------------
    # 11) Feature Importance
    # ----------------------------
    try:
        feature_names = (
            numeric_features + 
            list(clf.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(categorical_features))
        )
        importances = clf.named_steps['classifier'].feature_importances_
        
        # Get top 20 features
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importances[indices][::-1])
        plt.yticks(range(20), [feature_names[i] for i in indices[::-1]])
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importances for Readmission Prediction")
        plt.tight_layout()
        plt.savefig("readmission_feature_importance.png", bbox_inches="tight")
        plt.close()
        print("Feature importance plot saved as readmission_feature_importance.png")
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, idx in enumerate(indices[:10], 1):
            print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
            
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    # ----------------------------
    # 12) High-Risk Patient Analysis
    # ----------------------------
    high_risk_threshold = 0.5
    high_risk_patients = sum(y_prob >= high_risk_threshold)
    actual_readmissions_in_high_risk = sum((y_prob >= high_risk_threshold) & (y_test == 1))
    
    print(f"\n--- High-Risk Analysis (threshold = {high_risk_threshold}) ---")
    print(f"Patients flagged as high-risk: {high_risk_patients}")
    print(f"Actual readmissions in high-risk group: {actual_readmissions_in_high_risk}")
    if high_risk_patients > 0:
        print(f"Precision in high-risk group: {actual_readmissions_in_high_risk / high_risk_patients:.2%}")


if __name__ == "__main__":
    main()
