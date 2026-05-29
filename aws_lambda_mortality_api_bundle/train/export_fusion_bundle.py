from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Ensure the project root is on sys.path so 'common' is importable
# regardless of whether the script is invoked as `python train/export_fusion_bundle.py`
# (which only adds train/ to sys.path) or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common.features import LOS_NUMERIC_FEATURES, build_training_features


REQUIRED_ADMISSIONS_COLS = [
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
]

REQUIRED_PATIENTS_COLS = ["subject_id", "gender", "anchor_age", "anchor_year"]
REQUIRED_CODE_COLS = ["hadm_id", "icd_code"]
REQUIRED_RX_COLS = ["hadm_id", "drug"]


def safe_read_csv(path: str, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, dtype=dtype, low_memory=False)


def read_prescriptions_aggregated(path: str, chunksize: int = 200_000) -> pd.DataFrame:
    """Read a large prescriptions CSV in chunks and return only hadm_id + drug,
    deduplicated per chunk to keep memory low."""
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=["hadm_id", "drug"],
        dtype={"hadm_id": "Int32", "drug": "string"},
        low_memory=False,
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int32")
        # Drop duplicate (hadm_id, drug) pairs within this chunk to cut size early
        chunk = chunk.drop_duplicates()
        parts.append(chunk)
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    return result


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(numeric_features, categorical_features, min_class_count: int) -> Dict[str, Pipeline]:
    cal_cv = 3 if min_class_count >= 3 else 2

    def make_pipe(classifier):
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("classifier", classifier),
            ]
        )

    svm = make_pipe(
        CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000),
            cv=cal_cv,
        )
    )
    dt = make_pipe(DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42))
    rf = make_pipe(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    )
    gb = make_pipe(GradientBoostingClassifier(random_state=42))
    lr = make_pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))

    return {
        "svm_calibrated": svm,
        "decision_tree": dt,
        "random_forest": rf,
        "gradient_boosting": gb,
        "logistic_regression": lr,
    }


def fit_and_report(models: Dict[str, Pipeline], X_train, X_test, y_train, y_test) -> None:
    print("\n=== Holdout sanity-check metrics ===")
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        line = f"{name}: accuracy={acc:.4f}"
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, prob)
                line += f", auc={auc:.4f}"
            except Exception:
                pass
        print(line)


def refit_full(models: Dict[str, Pipeline], X, y) -> Dict[str, Pipeline]:
    trained: Dict[str, Pipeline] = {}
    print("\n=== Refit on full dataset for deployment ===")
    for name, model in models.items():
        print(f"Refitting {name} on full dataset ...")
        model.fit(X, y)
        trained[name] = model
    return trained


def build_los_training_data(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    admissions = admissions.copy()
    admissions = admissions.dropna(subset=["dischtime", "admittime"])
    admissions["LOS_DAYS"] = (
        (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")
    admissions = admissions[(admissions["LOS_DAYS"] > 0) & (admissions["LOS_DAYS"] <= 365)]

    admissions["ADMIT_HOUR"] = admissions["admittime"].dt.hour.astype("int32")
    admissions["ADMIT_DAY_OF_WEEK"] = admissions["admittime"].dt.dayofweek.astype("int32")
    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year.astype("int32")

    base = admissions.merge(patients, on="subject_id", how="left")
    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")
    base["IS_MALE"] = (base["gender"] == "M").astype("int8")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["ED_LOS_HOURS"] = (
        (base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600
    ).fillna(0).astype("float32")
    base["IS_EMERGENCY"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["HAS_MEDICARE"] = (base["insurance"].str.lower() == "medicare").astype("int8")
    base["HAS_MEDICAID"] = (base["insurance"].str.lower() == "medicaid").astype("int8")

    diagnoses_c = diagnoses.dropna(subset=["hadm_id"]).copy()
    diagnoses_c["hadm_id"] = diagnoses_c["hadm_id"].astype("int32")
    procedures_c = procedures.dropna(subset=["hadm_id"]).copy()
    procedures_c["hadm_id"] = procedures_c["hadm_id"].astype("int32")

    num_conditions = diagnoses_c.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_procedures = procedures_c.groupby("hadm_id").size().rename("NUM_PROCEDURES")

    feat = (
        base.set_index("hadm_id")
        .join(num_conditions, how="left")
        .join(num_procedures, how="left")
        .reset_index()
    )
    feat["NUM_CONDITIONS"] = feat["NUM_CONDITIONS"].fillna(0).astype("int32")
    feat["NUM_PROCEDURES"] = feat["NUM_PROCEDURES"].fillna(0).astype("int32")

    data = feat[LOS_NUMERIC_FEATURES + ["LOS_DAYS"]].dropna(subset=["AGE"])
    X = data[LOS_NUMERIC_FEATURES]
    y = data["LOS_DAYS"]
    return X, y


def train_los_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    print("\n=== Training LOS model ===")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ])
    pipeline.fit(X.values, y.values)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    pipeline_eval = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ])
    pipeline_eval.fit(X_train, y_train)
    preds = pipeline_eval.predict(X_test)
    import numpy as np
    mae = float(np.mean(np.abs(y_test - preds)))
    print(f"LOS holdout MAE: {mae:.3f} days")
    print("Refitting LOS model on full dataset ...")
    return pipeline


def build_readmission_training_data(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    prescriptions: pd.DataFrame | None,
    top_n_icd_groups: int = 15,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str], bool]:
    admissions = admissions.copy()
    admissions = admissions.dropna(subset=["dischtime", "admittime"])
    # Exclude deaths — they cannot be readmitted
    admissions = admissions[admissions["hospital_expire_flag"] == 0]
    admissions = admissions.sort_values(["subject_id", "dischtime"]).reset_index(drop=True)

    admissions["LOS_DAYS"] = (
        (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")

    admissions["next_admittime"] = admissions.groupby("subject_id")["admittime"].shift(-1)
    admissions["days_to_readmission"] = (
        (admissions["next_admittime"] - admissions["dischtime"]).dt.total_seconds() / 86400
    )
    admissions["READMITTED_30D"] = (
        (admissions["days_to_readmission"] <= 30) & (admissions["days_to_readmission"] > 0)
    ).astype(int)
    admissions = admissions[admissions["next_admittime"].notna()].copy()

    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year.astype("int32")
    base = admissions.merge(patients, on="subject_id", how="left")

    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["ED_LOS_HOURS"] = (
        (base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600
    ).fillna(0).astype("float32")
    base["EMERGENCY_ADMISSION"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["ELECTIVE_ADMISSION"] = (base["admission_type"] == "ELECTIVE").astype("int8")
    base["GENDER"] = base["gender"]
    base["RACE"] = base["race"]
    base["LANGUAGE"] = base["language"]
    base["DISCHARGE_LOCATION"] = base["discharge_location"] if "discharge_location" in base.columns else None

    base["admission_number"] = base.groupby("subject_id").cumcount() + 1
    base["PREVIOUS_ADMISSIONS"] = base["admission_number"] - 1
    base["prev_dischtime"] = base.groupby("subject_id")["dischtime"].shift(1)
    base["DAYS_SINCE_LAST_ADMISSION"] = (
        (base["admittime"] - base["prev_dischtime"]).dt.total_seconds() / 86400
    ).fillna(0).astype("float32")

    diagnoses_c = diagnoses.dropna(subset=["hadm_id"]).copy()
    diagnoses_c["hadm_id"] = diagnoses_c["hadm_id"].astype("int32")
    procedures_c = procedures.dropna(subset=["hadm_id"]).copy()
    procedures_c["hadm_id"] = procedures_c["hadm_id"].astype("int32")

    num_conditions = diagnoses_c.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_unique_conditions = diagnoses_c.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_CONDITIONS")
    num_procedures = procedures_c.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    num_unique_procedures = procedures_c.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_PROCEDURES")

    diagnoses_c["ICD_GROUP"] = diagnoses_c["icd_code"].astype("string").str[0].fillna("?")
    icd_groups = diagnoses_c["ICD_GROUP"].value_counts().nlargest(top_n_icd_groups).index.tolist()
    dx_top = (
        diagnoses_c[diagnoses_c["ICD_GROUP"].isin(icd_groups)]
        .groupby(["hadm_id", "ICD_GROUP"])
        .size()
        .unstack(fill_value=0)
    )
    dx_top.columns = [f"ICD_{c}_COUNT" for c in dx_top.columns]
    icd_cols = list(dx_top.columns)

    feat = (
        base.set_index("hadm_id")
        .join(num_conditions, how="left")
        .join(num_unique_conditions, how="left")
        .join(num_procedures, how="left")
        .join(num_unique_procedures, how="left")
        .join(dx_top, how="left")
    )

    has_rx_features = prescriptions is not None
    if prescriptions is not None:
        prescriptions_c = prescriptions.dropna(subset=["hadm_id"]).copy()
        prescriptions_c["hadm_id"] = prescriptions_c["hadm_id"].astype("int32")
        num_meds = prescriptions_c.groupby("hadm_id").size().rename("NUM_MEDICATIONS")
        num_unique_meds = prescriptions_c.groupby("hadm_id")["drug"].nunique().rename("NUM_UNIQUE_MEDICATIONS")
        feat = feat.join(num_meds, how="left").join(num_unique_meds, how="left")

    feat = feat.reset_index()

    count_cols = ["NUM_CONDITIONS", "NUM_UNIQUE_CONDITIONS", "NUM_PROCEDURES", "NUM_UNIQUE_PROCEDURES"]
    if has_rx_features:
        count_cols += ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]
    for c in count_cols + icd_cols:
        feat[c] = feat[c].fillna(0).astype("int32")

    numeric_features = [
        "AGE", "HAS_ED_VISIT", "ED_LOS_HOURS", "EMERGENCY_ADMISSION", "ELECTIVE_ADMISSION",
        "LOS_DAYS", "PREVIOUS_ADMISSIONS", "DAYS_SINCE_LAST_ADMISSION",
        "NUM_CONDITIONS", "NUM_UNIQUE_CONDITIONS", "NUM_PROCEDURES", "NUM_UNIQUE_PROCEDURES",
    ] + icd_cols
    if has_rx_features:
        numeric_features += ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]

    categorical_features = [
        "GENDER", "RACE", "LANGUAGE",
        "admission_type", "admission_location", "DISCHARGE_LOCATION",
        "insurance", "marital_status",
    ]

    if "DISCHARGE_LOCATION" not in feat.columns:
        feat["DISCHARGE_LOCATION"] = None

    target = "READMITTED_30D"
    data = feat[numeric_features + categorical_features + [target]].dropna(
        subset=["AGE", "GENDER", "RACE", "LOS_DAYS", target]
    )

    X = data[numeric_features + categorical_features]
    y = data[target].astype(int)
    return X, y, numeric_features, categorical_features, icd_groups, has_rx_features


def train_readmission_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    print("\n=== Training 30-day readmission model ===")
    print(f"Dataset: {X.shape[0]:,} samples, readmission rate: {y.mean():.2%}")

    value_counts = y.value_counts()
    if len(value_counts) < 2 or int(value_counts.min()) < 2:
        raise ValueError("Not enough samples in minority class for readmission model training.")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if int(value_counts.min()) >= 2 else None,
    )
    eval_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )),
    ])
    eval_pipeline.fit(X_train, y_train)
    preds = eval_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    try:
        probs = eval_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"Readmission holdout: accuracy={acc:.4f}, auc={auc:.4f}")
    except Exception:
        print(f"Readmission holdout: accuracy={acc:.4f}")

    print("Refitting readmission model on full dataset ...")
    pipeline.fit(X, y)
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a 5-model mortality fusion bundle.")
    parser.add_argument("--data-dir", required=True, help="Directory containing admissions/patients/diagnoses/procedures CSVs.")
    parser.add_argument("--output", default="model_artifacts/fusion_bundle.joblib", help="Where to save the joblib bundle.")
    parser.add_argument("--admissions", default="admissions.csv")
    parser.add_argument("--patients", default="patients.csv")
    parser.add_argument("--diagnoses", default="diagnoses_icd.csv")
    parser.add_argument("--procedures", default="procedures_icd.csv")
    parser.add_argument("--prescriptions", default="prescriptions.csv")
    parser.add_argument("--top-icd-groups", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    admissions_path = os.path.join(args.data_dir, args.admissions)
    patients_path = os.path.join(args.data_dir, args.patients)
    diagnoses_path = os.path.join(args.data_dir, args.diagnoses)
    procedures_path = os.path.join(args.data_dir, args.procedures)
    prescriptions_path = os.path.join(args.data_dir, args.prescriptions)

    admissions = safe_read_csv(
        admissions_path,
        usecols=REQUIRED_ADMISSIONS_COLS,
        parse_dates=["admittime", "dischtime", "edregtime", "edouttime"],
        dtype={"subject_id": "int32", "hadm_id": "int32"},
    )
    patients = safe_read_csv(
        patients_path,
        usecols=REQUIRED_PATIENTS_COLS,
        dtype={"subject_id": "int32", "anchor_age": "float32", "anchor_year": "int32"},
    )
    diagnoses = safe_read_csv(
        diagnoses_path,
        usecols=REQUIRED_CODE_COLS,
        dtype={"hadm_id": "Int32", "icd_code": "string"},
    )
    procedures = safe_read_csv(
        procedures_path,
        usecols=REQUIRED_CODE_COLS,
        dtype={"hadm_id": "Int32", "icd_code": "string"},
    )

    prescriptions = None
    if os.path.exists(prescriptions_path):
        print(f"Reading prescriptions in chunks (file may be large): {prescriptions_path}")
        prescriptions = read_prescriptions_aggregated(prescriptions_path)
        print(f"Prescriptions loaded: {len(prescriptions):,} unique (hadm_id, drug) pairs")
    else:
        print("No prescriptions file found. Training without medication features.")

    # ── Mortality models ──────────────────────────────────────────────────────
    data, numeric_features, categorical_features, icd_groups, has_rx_features = build_training_features(
        admissions=admissions,
        patients=patients,
        diagnoses=diagnoses,
        procedures=procedures,
        prescriptions=prescriptions,
        top_n_icd_groups=args.top_icd_groups,
    )

    X = data[numeric_features + categorical_features]
    y = data["MORTALITY"].astype(int)
    print(f"Mortality dataset: X={X.shape}, y={y.shape}")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Training data must contain both mortality classes.")

    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Need at least 2 rows in the minority class for calibrated SVM training.")

    stratify_labels = y if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify_labels,
    )

    models = build_models(numeric_features, categorical_features, int(y_train.value_counts().min()))
    fit_and_report(models, X_train, X_test, y_train, y_test)
    trained_models = refit_full(models, X, y)

    # ── LOS model ─────────────────────────────────────────────────────────────
    los_model = None
    try:
        X_los, y_los = build_los_training_data(admissions, patients, diagnoses, procedures)
        print(f"LOS dataset: {X_los.shape[0]:,} samples")
        los_model = train_los_model(X_los, y_los)
    except Exception as exc:
        print(f"WARNING: LOS model training failed: {exc}")

    # ── Readmission model ─────────────────────────────────────────────────────
    readmission_model = None
    readmission_numeric_features: List[str] = []
    readmission_categorical_features: List[str] = []
    readmission_icd_groups: List[str] = []
    readmission_has_rx_features = False
    try:
        X_ra, y_ra, ra_num, ra_cat, ra_icd, ra_has_rx = build_readmission_training_data(
            admissions, patients, diagnoses, procedures, prescriptions, args.top_icd_groups
        )
        print(f"Readmission dataset: {X_ra.shape[0]:,} samples")
        readmission_model = train_readmission_model(X_ra, y_ra, ra_num, ra_cat)
        readmission_numeric_features = ra_num
        readmission_categorical_features = ra_cat
        readmission_icd_groups = ra_icd
        readmission_has_rx_features = ra_has_rx
    except Exception as exc:
        print(f"WARNING: Readmission model training failed: {exc}")

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "models": trained_models,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "icd_groups": icd_groups,
        "has_rx_features": has_rx_features,
        # LOS
        "los_model": los_model,
        "los_feature_names": LOS_NUMERIC_FEATURES,
        # Readmission
        "readmission_model": readmission_model,
        "readmission_numeric_features": readmission_numeric_features,
        "readmission_categorical_features": readmission_categorical_features,
        "readmission_icd_groups": readmission_icd_groups,
        "readmission_has_rx_features": readmission_has_rx_features,
        "model_version": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"\nSaved bundle to: {args.output}")
    print(f"LOS model included: {los_model is not None}")
    print(f"Readmission model included: {readmission_model is not None}")


if __name__ == "__main__":
    main()
