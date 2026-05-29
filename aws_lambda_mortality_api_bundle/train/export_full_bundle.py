"""
Train all three prediction models and export a single combined bundle:
  - Mortality ensemble  (5 models: SVM, Decision Tree, Random Forest, GBM, Logistic)
  - Length-of-stay      (Ridge regression)
  - 30-day readmission  (Gradient Boosting classifier)

Usage:
  python train/export_full_bundle.py \\
    --data-dir data \\
    --output model_artifacts/fusion_bundle.joblib
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common.features import build_training_features, LOS_NUMERIC_FEATURES

RANDOM_STATE = 42

READMISSION_CATEGORICAL_FEATURES = [
    "GENDER",
    "RACE",
    "LANGUAGE",
    "admission_type",
    "admission_location",
    "DISCHARGE_LOCATION",
    "insurance",
    "marital_status",
]


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def safe_read_csv(path, usecols=None, parse_dates=None, dtype=None):
    return pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )


def _numeric_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _categorical_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipe(), numeric_features),
            ("cat", _categorical_pipe(), categorical_features),
        ]
    )


# ─────────────────────────────────────────────────────────────
# 1. MORTALITY — 5-model ensemble
# ─────────────────────────────────────────────────────────────

def train_mortality(admissions, patients, diagnoses, procedures, prescriptions,
                    top_n_icd, test_size):
    print("\n" + "=" * 60)
    print("MORTALITY MODEL — 5-model ensemble")
    print("=" * 60)

    data, num_feat, cat_feat, icd_groups, has_rx = build_training_features(
        admissions=admissions,
        patients=patients,
        diagnoses=diagnoses,
        procedures=procedures,
        prescriptions=prescriptions,
        top_n_icd_groups=top_n_icd,
    )

    X = data[num_feat + cat_feat]
    y = data["MORTALITY"].astype(int)
    print(f"  Dataset: {X.shape}, mortality rate: {y.mean():.2%}")

    min_class = int(y.value_counts().min())
    cal_cv = min(3, min_class)

    def make_pipe(clf):
        return Pipeline([
            ("preprocessor", build_preprocessor(num_feat, cat_feat)),
            ("classifier", clf),
        ])

    models = {
        "svm_calibrated": make_pipe(CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced",
                      random_state=RANDOM_STATE, max_iter=5000),
            cv=cal_cv,
        )),
        "decision_tree": make_pipe(
            DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                                   random_state=RANDOM_STATE)
        ),
        "random_forest": make_pipe(
            RandomForestClassifier(n_estimators=100, max_depth=20,
                                   min_samples_split=10, class_weight="balanced",
                                   n_jobs=-1, random_state=RANDOM_STATE)
        ),
        "gradient_boosting": make_pipe(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
        "logistic_regression": make_pipe(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE)
        ),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        line = f"  {name}: acc={acc:.4f}"
        try:
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
            line += f", auc={auc:.4f}"
        except Exception:
            pass
        print(line)

    print("  Refitting all models on full dataset...")
    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model

    return trained, num_feat, cat_feat, icd_groups, has_rx


# ─────────────────────────────────────────────────────────────
# 2. LENGTH OF STAY — Ridge regression
# ─────────────────────────────────────────────────────────────

def _build_los_features(admissions, patients, diagnoses, procedures):
    adm = admissions.copy()
    pat = patients.copy()
    dx = diagnoses.copy()
    px = procedures.copy()

    adm = adm.dropna(subset=["dischtime", "admittime"])
    adm["LOS_DAYS"] = (
        (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")
    adm = adm[(adm["LOS_DAYS"] > 0) & (adm["LOS_DAYS"] <= 365)]

    adm["ADMIT_HOUR"] = adm["admittime"].dt.hour.astype("int32")
    adm["ADMIT_DAY_OF_WEEK"] = adm["admittime"].dt.dayofweek.astype("int32")

    base = adm.merge(pat, on="subject_id", how="left")
    base["ADMIT_YEAR"] = base["admittime"].dt.year.astype("int32")
    base["AGE"] = (
        base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])
    ).astype("float32")
    base["ED_LOS_HOURS"] = (
        (base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600
    ).fillna(0).astype("float32")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["IS_EMERGENCY"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["IS_MALE"] = (base["gender"] == "M").astype("int8")
    base["HAS_MEDICARE"] = (base["insurance"] == "Medicare").astype("int8")
    base["HAS_MEDICAID"] = (base["insurance"] == "Medicaid").astype("int8")

    dx = dx.dropna(subset=["hadm_id"])
    dx["hadm_id"] = dx["hadm_id"].astype("int32")
    px = px.dropna(subset=["hadm_id"])
    px["hadm_id"] = px["hadm_id"].astype("int32")

    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")

    feat = (
        base.set_index("hadm_id")
        .join(num_conditions, how="left")
        .join(num_procedures, how="left")
        .reset_index()
    )
    feat["NUM_CONDITIONS"] = feat["NUM_CONDITIONS"].fillna(0).astype("int32")
    feat["NUM_PROCEDURES"] = feat["NUM_PROCEDURES"].fillna(0).astype("int32")

    return feat[LOS_NUMERIC_FEATURES + ["LOS_DAYS"]].dropna(subset=["AGE"])


def train_los(admissions, patients, diagnoses, procedures, test_size):
    print("\n" + "=" * 60)
    print("LENGTH-OF-STAY MODEL — Ridge regression")
    print("=" * 60)

    feat = _build_los_features(admissions, patients, diagnoses, procedures)
    X = feat[LOS_NUMERIC_FEATURES]
    y = feat["LOS_DAYS"].values
    print(f"  Dataset: {X.shape[0]:,} samples, mean LOS={y.mean():.2f} days")

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", _numeric_pipe(), LOS_NUMERIC_FEATURES),
        ])),
        ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    pipeline.fit(X_tr, y_tr)
    y_pred = np.maximum(pipeline.predict(X_te), 0)
    print(f"  Holdout MAE: {mean_absolute_error(y_te, y_pred):.3f} days")

    pipeline.fit(X, y)
    return pipeline


# ─────────────────────────────────────────────────────────────
# 3. 30-DAY READMISSION — Gradient Boosting classifier
# ─────────────────────────────────────────────────────────────

READMISSION_NUMERIC_BASE = [
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
]


def _build_readmission_features(admissions, patients, diagnoses, procedures,
                                 prescriptions, top_n_icd):
    adm = admissions.copy()
    pat = patients.copy()
    dx = diagnoses.copy()
    px = procedures.copy()

    # Remove in-hospital deaths (they cannot be readmitted)
    adm = adm.dropna(subset=["dischtime", "admittime"])
    adm = adm[adm["hospital_expire_flag"] == 0]
    adm = adm.sort_values(["subject_id", "dischtime"]).reset_index(drop=True)

    adm["LOS_DAYS"] = (
        (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    ).astype("float32")
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["days_to_readmission"] = (
        (adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() / 86400
    )
    adm["READMITTED_30D"] = (
        (adm["days_to_readmission"] <= 30) & (adm["days_to_readmission"] > 0)
    ).astype(int)
    # Drop last admission per patient (no follow-up known)
    adm = adm[adm["next_admittime"].notna()].copy()

    adm["ADMIT_YEAR"] = adm["admittime"].dt.year.astype("int32")
    base = adm.merge(pat, on="subject_id", how="left")
    base["AGE"] = (
        base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])
    ).astype("float32")
    base["ED_LOS_HOURS"] = (
        (base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600
    ).fillna(0).astype("float32")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["EMERGENCY_ADMISSION"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["ELECTIVE_ADMISSION"] = (base["admission_type"] == "ELECTIVE").astype("int8")
    base["GENDER"] = base["gender"]
    base["RACE"] = base["race"]
    base["LANGUAGE"] = base["language"]
    base["DISCHARGE_LOCATION"] = base["discharge_location"]

    base["admission_number"] = base.groupby("subject_id").cumcount() + 1
    base["PREVIOUS_ADMISSIONS"] = base["admission_number"] - 1
    base["prev_dischtime"] = base.groupby("subject_id")["dischtime"].shift(1)
    base["DAYS_SINCE_LAST_ADMISSION"] = (
        (base["admittime"] - base["prev_dischtime"]).dt.total_seconds() / 86400
    ).fillna(0).astype("float32")

    dx = dx.dropna(subset=["hadm_id"])
    dx["hadm_id"] = dx["hadm_id"].astype("int32")
    px = px.dropna(subset=["hadm_id"])
    px["hadm_id"] = px["hadm_id"].astype("int32")

    num_conditions = dx.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_unique_conditions = dx.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_CONDITIONS")
    num_procedures = px.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    num_unique_procedures = px.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_PROCEDURES")

    dx["ICD_GROUP"] = dx["icd_code"].astype("string").str[0].fillna("?")
    icd_groups = dx["ICD_GROUP"].value_counts().nlargest(top_n_icd).index.tolist()
    dx_top = (
        dx[dx["ICD_GROUP"].isin(icd_groups)]
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
        .reset_index()
    )

    has_rx = False
    if prescriptions is not None:
        rx = prescriptions.copy()
        rx = rx.dropna(subset=["hadm_id"])
        rx["hadm_id"] = rx["hadm_id"].astype("int32")
        num_meds = rx.groupby("hadm_id").size().rename("NUM_MEDICATIONS")
        num_unique_meds = rx.groupby("hadm_id")["drug"].nunique().rename("NUM_UNIQUE_MEDICATIONS")
        feat = (
            feat.set_index("hadm_id")
            .join(num_meds, how="left")
            .join(num_unique_meds, how="left")
            .reset_index()
        )
        feat["NUM_MEDICATIONS"] = feat["NUM_MEDICATIONS"].fillna(0).astype("int32")
        feat["NUM_UNIQUE_MEDICATIONS"] = feat["NUM_UNIQUE_MEDICATIONS"].fillna(0).astype("int32")
        has_rx = True

    for c in ["NUM_CONDITIONS", "NUM_UNIQUE_CONDITIONS",
              "NUM_PROCEDURES", "NUM_UNIQUE_PROCEDURES"] + icd_cols:
        feat[c] = feat[c].fillna(0).astype("int32")

    numeric_features = READMISSION_NUMERIC_BASE + icd_cols
    if has_rx:
        numeric_features += ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]

    print(f"  Readmission dataset: {len(adm):,} admissions with follow-up, "
          f"rate: {adm['READMITTED_30D'].mean():.2%}")

    return feat, numeric_features, READMISSION_CATEGORICAL_FEATURES, icd_groups, has_rx


def train_readmission(admissions, patients, diagnoses, procedures, prescriptions,
                      top_n_icd, test_size):
    print("\n" + "=" * 60)
    print("30-DAY READMISSION MODEL — Gradient Boosting")
    print("=" * 60)

    feat, num_feat, cat_feat, icd_groups, has_rx = _build_readmission_features(
        admissions, patients, diagnoses, procedures, prescriptions, top_n_icd
    )

    data = feat[num_feat + cat_feat + ["READMITTED_30D"]].dropna(
        subset=["AGE", "GENDER", "RACE", "LOS_DAYS", "READMITTED_30D"]
    )
    X = data[num_feat + cat_feat]
    y = data["READMITTED_30D"].astype(int)
    print(f"  Dataset: {X.shape}, readmission rate: {y.mean():.2%}")

    clf = Pipeline([
        ("preprocessor", build_preprocessor(num_feat, cat_feat)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE,
        )),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    line = f"  Holdout: acc={acc:.4f}"
    try:
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        line += f", auc={auc:.4f}"
    except Exception:
        pass
    print(line)

    clf.fit(X, y)
    return clf, num_feat, cat_feat, icd_groups, has_rx


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train mortality, LOS, and readmission models and export a combined bundle."
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory containing MIMIC CSV files.")
    p.add_argument("--output", default="model_artifacts/fusion_bundle.joblib",
                   help="Output path for the joblib bundle.")
    p.add_argument("--admissions", default="admissions.csv")
    p.add_argument("--patients", default="patients.csv")
    p.add_argument("--diagnoses", default="diagnoses_icd.csv")
    p.add_argument("--procedures", default="procedures_icd.csv")
    p.add_argument("--prescriptions", default="prescriptions.csv")
    p.add_argument("--top-icd-groups", type=int, default=15)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    d = args.data_dir

    print("Loading CSVs...")
    admissions = safe_read_csv(
        os.path.join(d, args.admissions),
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race",
            "edregtime", "edouttime", "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "edregtime", "edouttime"],
        dtype={"subject_id": "int32", "hadm_id": "int32"},
    )
    patients = safe_read_csv(
        os.path.join(d, args.patients),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        dtype={"subject_id": "int32", "anchor_age": "float32", "anchor_year": "int32"},
    )
    diagnoses = safe_read_csv(
        os.path.join(d, args.diagnoses),
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "Int32", "icd_code": "string"},
    )
    procedures = safe_read_csv(
        os.path.join(d, args.procedures),
        usecols=["hadm_id", "icd_code"],
        dtype={"hadm_id": "Int32", "icd_code": "string"},
    )

    prescriptions = None
    rx_path = os.path.join(d, args.prescriptions)
    if os.path.exists(rx_path):
        prescriptions = safe_read_csv(
            rx_path,
            usecols=["hadm_id", "drug"],
            dtype={"hadm_id": "Int32", "drug": "string"},
        )
        print(f"Prescriptions loaded: {prescriptions.shape}")
    else:
        print("No prescriptions file found. Skipping medication features.")

    # ── Train all three model groups ──────────────────────────
    mortality_models, num_feat, cat_feat, icd_groups, has_rx = train_mortality(
        admissions, patients, diagnoses, procedures, prescriptions,
        top_n_icd=args.top_icd_groups, test_size=args.test_size,
    )

    los_model = train_los(
        admissions, patients, diagnoses, procedures,
        test_size=args.test_size,
    )

    ra_model, ra_num_feat, ra_cat_feat, ra_icd_groups, ra_has_rx = train_readmission(
        admissions, patients, diagnoses, procedures, prescriptions,
        top_n_icd=args.top_icd_groups, test_size=args.test_size,
    )

    # ── Bundle ───────────────────────────────────────────────
    bundle = {
        # Mortality ensemble
        "models": mortality_models,
        "numeric_features": num_feat,
        "categorical_features": cat_feat,
        "icd_groups": icd_groups,
        "has_rx_features": has_rx,

        # Length-of-stay
        "los_model": los_model,
        "los_feature_names": LOS_NUMERIC_FEATURES,

        # 30-day readmission
        "readmission_model": ra_model,
        "readmission_numeric_features": ra_num_feat,
        "readmission_categorical_features": ra_cat_feat,
        "readmission_icd_groups": ra_icd_groups,
        "readmission_has_rx_features": ra_has_rx,

        "model_version": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"\nSaved combined bundle to: {args.output}")
    print(f"  Mortality models : {list(mortality_models.keys())}")
    print(f"  LOS features     : {LOS_NUMERIC_FEATURES}")
    print(f"  Readmission num  : {len(ra_num_feat)} features")
    print(f"  Readmission cat  : {len(ra_cat_feat)} features")


if __name__ == "__main__":
    main()
