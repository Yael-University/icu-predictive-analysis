import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "MIMIC"

ADMISSIONS_CSV = os.path.join(DATA_DIR, "admissions.csv")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
DIAGNOSES_CSV = os.path.join(DATA_DIR, "diagnoses_icd.csv")
PROCEDURES_CSV = os.path.join(DATA_DIR, "procedures_icd.csv")

# Optional (only used if present)
PRESCRIPTIONS_CSV = os.path.join(DATA_DIR, "prescriptions.csv")

RANDOM_STATE = 42


def must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def safe_read_csv(path: str, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
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
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "edregtime", "edouttime"],
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

    # 2) Base + label + age
    #     (Leakage removed: no dischtime, discharge_location, deathtime, LOS)
    admissions["MORTALITY"] = admissions["hospital_expire_flag"].astype(int)
    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year.astype("int32")

    base = admissions.merge(patients, on="subject_id", how="left")

    # Approx age at admission (MIMIC de-identified anchors)
    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")

    # ED length (hours) if ED timestamps exist for that admission
    base["ED_LOS_HOURS"] = ((base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600).astype("float32")

    # Flags: ED present
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")

    # Admission-type binaries
    base["EMERGENCY_ADMISSION"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["ELECTIVE_ADMISSION"] = (base["admission_type"] == "ELECTIVE").astype("int8")

    # Match categorical names
    base["GENDER"] = base["gender"]
    base["RACE"] = base["race"]
    base["LANGUAGE"] = base["language"]

    # 3) Feature engineering from dx/px/rx
    # Diagnoses counts
    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_unique_conditions = dx.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_CONDITIONS")

    # Procedures counts
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    num_unique_procedures = px.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_PROCEDURES")

    # ICD group features (first char bucket) - top 15 most common buckets
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

    # 4) Merge into one wide table (one row per hadm_id)
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

    # 5) Final feature set (leakage removed)
    numeric_features = [
        "AGE",
        "HAS_ED_VISIT",
        "ED_LOS_HOURS",
        "EMERGENCY_ADMISSION",
        "ELECTIVE_ADMISSION",
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
        "insurance",
        "marital_status",
    ]

    target = "MORTALITY"

    # Drop rows missing critical features
    data = feat[numeric_features + categorical_features + [target]].dropna(
        subset=["AGE", "GENDER", "RACE"]
    )

    X = data[numeric_features + categorical_features]
    y = data[target].astype(int)

    print("\nDataset ready:")
    print("  X:", X.shape, "y:", y.shape)
    print("  mortality counts:\n", y.value_counts())

    # 6) Preprocess + AdaBoost
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

    # Base estimator for AdaBoost
    base_tree = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    # AdaBoost classifier
    ada = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,
        learning_rate=1.0,
        random_state=RANDOM_STATE
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ada),
        ]
    )

    value_counts = y.value_counts()
    stratify_labels = y if len(value_counts) >= 2 and value_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    # 7) Train and Evaluate
    print("\nTraining AdaBoost...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n" + "=" * 80)
    print("AdaBoost Results (MIMIC-IV features, leakage removed)")
    print("Accuracy:", f"{accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 8) Save Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
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
    plt.title("Confusion Matrix - AdaBoost (MIMIC-IV)")
    plt.tight_layout()
    plt.savefig("adaboost_confusion_matrix_mimic.png", bbox_inches="tight")
    plt.close()

    print("\nConfusion matrix saved as adaboost_confusion_matrix_mimic.png")

if __name__ == "__main__":
    main()
