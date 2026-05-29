from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

LOS_NUMERIC_FEATURES = [
    "AGE",
    "IS_MALE",
    "HAS_ED_VISIT",
    "ED_LOS_HOURS",
    "IS_EMERGENCY",
    "HAS_MEDICARE",
    "HAS_MEDICAID",
    "NUM_CONDITIONS",
    "NUM_PROCEDURES",
    "ADMIT_HOUR",
    "ADMIT_DAY_OF_WEEK",
]

BASE_NUMERIC_FEATURES = [
    "AGE",
    "HAS_ED_VISIT",
    "ED_LOS_HOURS",
    "EMERGENCY_ADMISSION",
    "ELECTIVE_ADMISSION",
    "NUM_CONDITIONS",
    "NUM_UNIQUE_CONDITIONS",
    "NUM_PROCEDURES",
    "NUM_UNIQUE_PROCEDURES",
]

MED_FEATURES = ["NUM_MEDICATIONS", "NUM_UNIQUE_MEDICATIONS"]

BASE_CATEGORICAL_FEATURES = [
    "GENDER",
    "RACE",
    "LANGUAGE",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
]


def _norm_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _norm_upper(value: Any) -> str | None:
    text = _norm_string(value)
    return text.upper() if text is not None else None


def _unique_nonempty(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in values:
        text = _norm_upper(item)
        if text is None:
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def build_feature_lists(icd_groups: List[str], has_rx_features: bool) -> Tuple[List[str], List[str]]:
    icd_cols = [f"ICD_{group}_COUNT" for group in icd_groups]
    numeric = BASE_NUMERIC_FEATURES + icd_cols
    if has_rx_features:
        numeric += MED_FEATURES
    categorical = list(BASE_CATEGORICAL_FEATURES)
    return numeric, categorical


def build_training_features(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    prescriptions: pd.DataFrame | None = None,
    top_n_icd_groups: int = 15,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], bool]:
    admissions = admissions.copy()
    patients = patients.copy()
    diagnoses = diagnoses.copy()
    procedures = procedures.copy()
    prescriptions = prescriptions.copy() if prescriptions is not None else None

    # Drop rows with null hadm_id (e.g. MIMIC-IV v3+ ED-only rows) and normalize to int32
    # so that downstream groupby/join operations match the admissions index dtype.
    diagnoses = diagnoses.dropna(subset=["hadm_id"])
    diagnoses["hadm_id"] = diagnoses["hadm_id"].astype("int32")
    procedures = procedures.dropna(subset=["hadm_id"])
    procedures["hadm_id"] = procedures["hadm_id"].astype("int32")
    if prescriptions is not None:
        prescriptions = prescriptions.dropna(subset=["hadm_id"])
        prescriptions["hadm_id"] = prescriptions["hadm_id"].astype("int32")

    admissions["MORTALITY"] = admissions["hospital_expire_flag"].astype(int)
    admissions["ADMIT_YEAR"] = admissions["admittime"].dt.year.astype("int32")

    base = admissions.merge(patients, on="subject_id", how="left")
    base["AGE"] = (base["anchor_age"] + (base["ADMIT_YEAR"] - base["anchor_year"])).astype("float32")
    base["ED_LOS_HOURS"] = ((base["edouttime"] - base["edregtime"]).dt.total_seconds() / 3600).astype("float32")
    base["HAS_ED_VISIT"] = base["edregtime"].notna().astype("int8")
    base["EMERGENCY_ADMISSION"] = (base["admission_type"] == "EMERGENCY").astype("int8")
    base["ELECTIVE_ADMISSION"] = (base["admission_type"] == "ELECTIVE").astype("int8")
    base["GENDER"] = base["gender"]
    base["RACE"] = base["race"]
    base["LANGUAGE"] = base["language"]

    num_conditions = diagnoses.groupby("hadm_id").size().rename("NUM_CONDITIONS")
    num_unique_conditions = diagnoses.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_CONDITIONS")
    num_procedures = procedures.groupby("hadm_id").size().rename("NUM_PROCEDURES")
    num_unique_procedures = procedures.groupby("hadm_id")["icd_code"].nunique().rename("NUM_UNIQUE_PROCEDURES")

    diagnoses["ICD_GROUP"] = diagnoses["icd_code"].astype("string").str[0].fillna("?")
    icd_groups = diagnoses["ICD_GROUP"].value_counts().nlargest(top_n_icd_groups).index.tolist()
    dx_top = (
        diagnoses[diagnoses["ICD_GROUP"].isin(icd_groups)]
        .groupby(["hadm_id", "ICD_GROUP"])
        .size()
        .unstack(fill_value=0)
    )
    dx_top.columns = [f"ICD_{c}_COUNT" for c in dx_top.columns]

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
        num_meds = prescriptions.groupby("hadm_id").size().rename("NUM_MEDICATIONS")
        num_unique_meds = prescriptions.groupby("hadm_id")["drug"].nunique().rename("NUM_UNIQUE_MEDICATIONS")
        feat = feat.join(num_meds, how="left").join(num_unique_meds, how="left")

    feat = feat.reset_index().rename(columns={"hadm_id": "Id"})

    count_cols = [
        "NUM_CONDITIONS",
        "NUM_UNIQUE_CONDITIONS",
        "NUM_PROCEDURES",
        "NUM_UNIQUE_PROCEDURES",
    ]
    if has_rx_features:
        count_cols += MED_FEATURES

    for col in count_cols:
        feat[col] = feat[col].fillna(0).astype("int32")

    icd_cols = [c for c in feat.columns if c.startswith("ICD_") and c.endswith("_COUNT")]
    for col in icd_cols:
        feat[col] = feat[col].fillna(0).astype("int32")

    numeric_features = BASE_NUMERIC_FEATURES + icd_cols
    if has_rx_features:
        numeric_features += MED_FEATURES
    categorical_features = list(BASE_CATEGORICAL_FEATURES)

    data = feat[numeric_features + categorical_features + ["MORTALITY"]].dropna(subset=["AGE", "GENDER", "RACE"])
    return data, numeric_features, categorical_features, icd_groups, has_rx_features


def patient_payload_to_row(patient_payload: Dict[str, Any], bundle: Dict[str, Any]) -> pd.DataFrame:
    numeric_features: List[str] = bundle["numeric_features"]
    categorical_features: List[str] = bundle["categorical_features"]
    icd_groups: List[str] = bundle["icd_groups"]
    has_rx_features: bool = bundle["has_rx_features"]

    if "flat_features" in patient_payload:
        raw = dict(patient_payload["flat_features"] or {})
        row = {feature: raw.get(feature) for feature in numeric_features + categorical_features}
        for feature in numeric_features:
            if feature.startswith("ICD_") and feature.endswith("_COUNT") and row.get(feature) is None:
                row[feature] = 0
        if has_rx_features:
            for feature in MED_FEATURES:
                row[feature] = raw.get(feature, row.get(feature, 0))
        return pd.DataFrame([row], columns=numeric_features + categorical_features)

    demographics = patient_payload.get("demographics") or {}
    admission = patient_payload.get("admission") or {}
    diagnoses = patient_payload.get("diagnoses") or []
    procedures = patient_payload.get("procedures") or []
    medications = patient_payload.get("medications") or []

    diagnoses_clean = [_norm_upper(code) for code in diagnoses if _norm_upper(code) is not None]
    procedures_clean = [_norm_upper(code) for code in procedures if _norm_upper(code) is not None]
    medications_clean = [_norm_string(drug) for drug in medications if _norm_string(drug) is not None]

    row: Dict[str, Any] = {
        "AGE": demographics.get("age"),
        "HAS_ED_VISIT": 1 if admission.get("has_ed_visit") else 0,
        "ED_LOS_HOURS": admission.get("ed_los_hours"),
        "EMERGENCY_ADMISSION": 1 if _norm_upper(admission.get("admission_type")) == "EMERGENCY" else 0,
        "ELECTIVE_ADMISSION": 1 if _norm_upper(admission.get("admission_type")) == "ELECTIVE" else 0,
        "NUM_CONDITIONS": len(diagnoses_clean),
        "NUM_UNIQUE_CONDITIONS": len(_unique_nonempty(diagnoses_clean)),
        "NUM_PROCEDURES": len(procedures_clean),
        "NUM_UNIQUE_PROCEDURES": len(_unique_nonempty(procedures_clean)),
        "GENDER": demographics.get("gender"),
        "RACE": demographics.get("race"),
        "LANGUAGE": demographics.get("language"),
        "admission_type": admission.get("admission_type"),
        "admission_location": admission.get("admission_location"),
        "insurance": admission.get("insurance"),
        "marital_status": admission.get("marital_status"),
    }

    if has_rx_features:
        row["NUM_MEDICATIONS"] = len(medications_clean)
        row["NUM_UNIQUE_MEDICATIONS"] = len(_unique_nonempty(medications_clean))

    for group in icd_groups:
        row[f"ICD_{group}_COUNT"] = 0

    for code in diagnoses_clean:
        group = code[0]
        col = f"ICD_{group}_COUNT"
        if col in row:
            row[col] += 1

    for feature in numeric_features:
        row.setdefault(feature, 0)
    for feature in categorical_features:
        row.setdefault(feature, None)

    return pd.DataFrame([row], columns=numeric_features + categorical_features)


def patient_payload_to_los_row(patient_payload: Dict[str, Any], bundle: Dict[str, Any]) -> pd.DataFrame:
    """Convert a patient JSON payload to a feature row for the LOS Ridge model."""
    feature_names: List[str] = bundle["los_feature_names"]

    demographics = patient_payload.get("demographics") or {}
    admission = patient_payload.get("admission") or {}
    diagnoses = patient_payload.get("diagnoses") or []
    procedures = patient_payload.get("procedures") or []

    gender = _norm_upper(demographics.get("gender"))
    insurance = (_norm_string(admission.get("insurance")) or "").lower()

    diagnoses_clean = [c for c in (_norm_upper(x) for x in diagnoses) if c is not None]
    procedures_clean = [c for c in (_norm_upper(x) for x in procedures) if c is not None]

    row: Dict[str, Any] = {
        "AGE": demographics.get("age"),
        "IS_MALE": 1 if gender == "M" else 0,
        "HAS_ED_VISIT": 1 if admission.get("has_ed_visit") else 0,
        "ED_LOS_HOURS": admission.get("ed_los_hours") or 0.0,
        "IS_EMERGENCY": 1 if _norm_upper(admission.get("admission_type")) == "EMERGENCY" else 0,
        "HAS_MEDICARE": 1 if insurance == "medicare" else 0,
        "HAS_MEDICAID": 1 if insurance == "medicaid" else 0,
        "NUM_CONDITIONS": len(diagnoses_clean),
        "NUM_PROCEDURES": len(procedures_clean),
        # Optional timing fields — default to noon on Monday when not supplied
        "ADMIT_HOUR": admission.get("admit_hour", 12),
        "ADMIT_DAY_OF_WEEK": admission.get("admit_day_of_week", 0),
    }

    return pd.DataFrame([row], columns=feature_names)


def patient_payload_to_readmission_row(patient_payload: Dict[str, Any], bundle: Dict[str, Any]) -> pd.DataFrame:
    """Convert a patient JSON payload to a feature row for the 30-day readmission model."""
    numeric_features: List[str] = bundle["readmission_numeric_features"]
    categorical_features: List[str] = bundle["readmission_categorical_features"]
    icd_groups: List[str] = bundle["readmission_icd_groups"]
    has_rx_features: bool = bundle["readmission_has_rx_features"]

    demographics = patient_payload.get("demographics") or {}
    admission = patient_payload.get("admission") or {}
    diagnoses = patient_payload.get("diagnoses") or []
    procedures = patient_payload.get("procedures") or []
    medications = patient_payload.get("medications") or []

    diagnoses_clean = [c for c in (_norm_upper(x) for x in diagnoses) if c is not None]
    procedures_clean = [c for c in (_norm_upper(x) for x in procedures) if c is not None]
    medications_clean = [c for c in (_norm_string(x) for x in medications) if c is not None]

    row: Dict[str, Any] = {
        "AGE": demographics.get("age"),
        "HAS_ED_VISIT": 1 if admission.get("has_ed_visit") else 0,
        "ED_LOS_HOURS": admission.get("ed_los_hours") or 0.0,
        "EMERGENCY_ADMISSION": 1 if _norm_upper(admission.get("admission_type")) == "EMERGENCY" else 0,
        "ELECTIVE_ADMISSION": 1 if _norm_upper(admission.get("admission_type")) == "ELECTIVE" else 0,
        # Known at discharge; at admission time these default to 0/None
        "LOS_DAYS": admission.get("los_days", 0.0),
        "PREVIOUS_ADMISSIONS": admission.get("previous_admissions", 0),
        "DAYS_SINCE_LAST_ADMISSION": admission.get("days_since_last_admission", 0.0),
        "NUM_CONDITIONS": len(diagnoses_clean),
        "NUM_UNIQUE_CONDITIONS": len(_unique_nonempty(diagnoses_clean)),
        "NUM_PROCEDURES": len(procedures_clean),
        "NUM_UNIQUE_PROCEDURES": len(_unique_nonempty(procedures_clean)),
        "GENDER": demographics.get("gender"),
        "RACE": demographics.get("race"),
        "LANGUAGE": demographics.get("language"),
        "admission_type": admission.get("admission_type"),
        "admission_location": admission.get("admission_location"),
        "DISCHARGE_LOCATION": admission.get("discharge_location"),
        "insurance": admission.get("insurance"),
        "marital_status": admission.get("marital_status"),
    }

    if has_rx_features:
        row["NUM_MEDICATIONS"] = len(medications_clean)
        row["NUM_UNIQUE_MEDICATIONS"] = len(_unique_nonempty(medications_clean))

    for group in icd_groups:
        row[f"ICD_{group}_COUNT"] = 0
    for code in diagnoses_clean:
        col = f"ICD_{code[0]}_COUNT"
        if col in row:
            row[col] += 1

    for feature in numeric_features:
        row.setdefault(feature, 0)
    for feature in categorical_features:
        row.setdefault(feature, None)

    return pd.DataFrame([row], columns=numeric_features + categorical_features)
