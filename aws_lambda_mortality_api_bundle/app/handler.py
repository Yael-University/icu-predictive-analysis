from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List

import joblib

from common.features import (
    patient_payload_to_row,
    patient_payload_to_los_row,
    patient_payload_to_readmission_row,
)

_MODEL_BUNDLE: Dict[str, Any] | None = None

_FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "AGE": "Age",
    "HAS_ED_VISIT": "Had ED Visit",
    "ED_LOS_HOURS": "ED LOS (hrs)",
    "EMERGENCY_ADMISSION": "Emergency Admission",
    "ELECTIVE_ADMISSION": "Elective Admission",
    "NUM_CONDITIONS": "# Diagnoses",
    "NUM_UNIQUE_CONDITIONS": "# Unique Diagnoses",
    "NUM_PROCEDURES": "# Procedures",
    "NUM_UNIQUE_PROCEDURES": "# Unique Procedures",
    "NUM_MEDICATIONS": "# Medications",
    "NUM_UNIQUE_MEDICATIONS": "# Unique Meds",
    "IS_MALE": "Is Male",
    "IS_EMERGENCY": "Emergency Admission",
    "HAS_MEDICARE": "Has Medicare",
    "HAS_MEDICAID": "Has Medicaid",
    "ADMIT_HOUR": "Admission Hour",
    "ADMIT_DAY_OF_WEEK": "Admission Day",
    "LOS_DAYS": "LOS (days)",
    "PREVIOUS_ADMISSIONS": "Prior Admissions",
    "DAYS_SINCE_LAST_ADMISSION": "Days Since Last Admit",
    "DISCHARGE_LOCATION": "Discharge Location",
}


def _display_name(feature: str) -> str:
    if feature in _FEATURE_DISPLAY_NAMES:
        return _FEATURE_DISPLAY_NAMES[feature]
    return feature.replace("_", " ").title()


def _cors_headers() -> Dict[str, str]:
    origin = os.environ.get("ALLOWED_ORIGIN", "*")
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "content-type,x-api-key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Content-Type": "application/json",
    }


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": _cors_headers(),
        "body": json.dumps(body, default=str),
    }


def _method(event: Dict[str, Any]) -> str:
    return ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET").upper()


def _path(event: Dict[str, Any]) -> str:
    return event.get("rawPath") or "/"


def _headers_lower(event: Dict[str, Any]) -> Dict[str, str]:
    headers = event.get("headers") or {}
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        return {}
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    if isinstance(body, (dict, list)):
        return body
    return json.loads(body)


def _check_api_key(event: Dict[str, Any]) -> bool:
    required = os.environ.get("API_KEY", "")
    if not required:
        return True
    headers = _headers_lower(event)
    return headers.get("x-api-key") == required


def _load_bundle() -> Dict[str, Any]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        model_path = os.environ.get("MODEL_PATH", "/var/task/model_artifacts/fusion_bundle.joblib")
        _MODEL_BUNDLE = joblib.load(model_path)
    return _MODEL_BUNDLE


def _risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.30:
        return "MODERATE"
    return "LOW"


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _get_transformer_feature_names(
    preprocessor, numeric_features: List[str], categorical_features: List[str]
) -> List[str]:
    try:
        ohe = preprocessor.transformers_[1][1].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = []
    return list(numeric_features) + cat_names


def _format_shap_pairs(
    feature_names: List[str], shap_values: Any, top_n: int = 15
) -> List[Dict[str, Any]]:
    pairs = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )[:top_n]
    return [
        {
            "feature": name,
            "display_name": _display_name(name),
            "value": round(float(val), 6),
        }
        for name, val in pairs
    ]


def _extract_tree_shap(explainer: Any, X_transformed: Any) -> Any:
    import numpy as np

    sv = explainer.shap_values(X_transformed)
    # Binary RF → list of [class0_array, class1_array], each (1, n_features)
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1][0]
    sv = np.asarray(sv)
    if sv.ndim == 3:   # (1, n_features, n_classes) — take class 1
        return sv[0, :, 1]
    if sv.ndim == 2:   # (1, n_features)
        return sv[0]
    return sv


def _compute_mortality_shap(bundle: Dict[str, Any], row_df: Any) -> List[Dict[str, Any]]:
    try:
        import shap
        rf_pipeline = bundle["models"]["random_forest"]
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        rf = rf_pipeline.named_steps["classifier"]
        all_names = _get_transformer_feature_names(
            preprocessor, bundle["numeric_features"], bundle["categorical_features"]
        )
        X_t = preprocessor.transform(row_df)
        sv = _extract_tree_shap(shap.TreeExplainer(rf), X_t)
        return _format_shap_pairs(all_names, sv)
    except Exception:
        return []


def _compute_los_shap(bundle: Dict[str, Any], los_row: Any) -> List[Dict[str, Any]]:
    try:
        los_model = bundle.get("los_model")
        if los_model is None:
            return []
        feature_names = bundle["los_feature_names"]
        imputer = los_model.named_steps["imputer"]
        scaler = los_model.named_steps["scaler"]
        ridge = los_model.named_steps["regressor"]
        X_imp = imputer.transform(los_row.values.astype(float))
        X_sc = scaler.transform(X_imp)
        sv = ridge.coef_ * X_sc[0]
        return _format_shap_pairs(feature_names, sv)
    except Exception:
        return []


def _compute_readmission_shap(bundle: Dict[str, Any], ra_row: Any) -> List[Dict[str, Any]]:
    try:
        import shap
        ra_model = bundle.get("readmission_model")
        if ra_model is None:
            return []
        preprocessor = ra_model.named_steps["preprocessor"]
        gbm = ra_model.named_steps["classifier"]
        all_names = _get_transformer_feature_names(
            preprocessor,
            bundle["readmission_numeric_features"],
            bundle["readmission_categorical_features"],
        )
        X_t = preprocessor.transform(ra_row)
        sv = _extract_tree_shap(shap.TreeExplainer(gbm), X_t)
        return _format_shap_pairs(all_names, sv)
    except Exception:
        return []


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _call_gemini(patient: Dict[str, Any], prediction: Dict[str, Any], api_key: str) -> str:
    mortality = prediction.get("mortality") or {}
    los = prediction.get("length_of_stay") or {}
    readmission = prediction.get("readmission_30d") or {}
    demographics = patient.get("demographics") or {}
    admission = patient.get("admission") or {}
    diagnoses = ", ".join(patient.get("diagnoses") or []) or "none provided"
    procedures = ", ".join(patient.get("procedures") or []) or "none provided"
    medications = ", ".join(patient.get("medications") or []) or "none provided"

    prompt = (
        "You are a clinical decision support assistant. Based on the patient data and ML risk "
        "predictions below, provide exactly 3 to 4 concise, actionable bullet points for the "
        "clinical team on how to best treat and manage this patient. Each bullet should be one "
        "focused, specific sentence starting with '•'.\n\n"
        "Patient Demographics:\n"
        f"- Age: {demographics.get('age')} | Gender: {demographics.get('gender')} | "
        f"Race: {demographics.get('race')} | Language: {demographics.get('language')}\n\n"
        "Admission:\n"
        f"- Type: {admission.get('admission_type')} | Insurance: {admission.get('insurance')} | "
        f"Location: {admission.get('admission_location')}\n\n"
        f"Diagnoses (ICD codes): {diagnoses}\n"
        f"Procedures: {procedures}\n"
        f"Medications: {medications}\n\n"
        "ML Risk Predictions:\n"
        f"- In-hospital Mortality: {mortality.get('risk_level')} "
        f"({mortality.get('death_percentage')}% probability)\n"
        f"- Predicted Length of Stay: {los.get('predicted_los_days')} days\n"
        f"- 30-Day Readmission: {readmission.get('risk_level')} "
        f"({readmission.get('readmission_percentage')}% probability)\n\n"
        "Respond with exactly 3-4 bullet points starting with '•'. Be specific and actionable."
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["candidates"][0]["content"]["parts"][0]["text"]


# ── Prediction ────────────────────────────────────────────────────────────────

def _predict(patient: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _load_bundle()

    # Compute feature rows upfront so SHAP reuses the same inputs
    row = patient_payload_to_row(patient, bundle)

    los_row = None
    if "los_model" in bundle and "los_feature_names" in bundle:
        try:
            los_row = patient_payload_to_los_row(patient, bundle)
        except Exception:
            pass

    ra_row = None
    if "readmission_model" in bundle and "readmission_numeric_features" in bundle:
        try:
            ra_row = patient_payload_to_readmission_row(patient, bundle)
        except Exception:
            pass

    # ── Mortality ensemble ────────────────────────────────────
    votes: Dict[str, int] = {}
    probabilities: Dict[str, float] = {}
    for model_name, model in bundle["models"].items():
        pred = int(model.predict(row)[0])
        votes[model_name] = pred
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(row)[0][1])
        else:
            prob = float(pred)
        probabilities[model_name] = prob

    vote_sum = sum(votes.values())
    model_count = len(votes)
    death_probability = sum(probabilities.values()) / model_count if model_count else 0.0
    predicted_class = 1 if vote_sum >= (model_count // 2 + 1) else 0

    # ── Length of Stay ────────────────────────────────────────
    los_result: Dict[str, Any] = {"predicted_los_days": None}
    if los_row is not None:
        try:
            raw_los = float(bundle["los_model"].predict(los_row)[0])
            los_result = {"predicted_los_days": round(max(raw_los, 0.0), 2)}
        except Exception as exc:
            los_result = {"predicted_los_days": None, "error": str(exc)}

    # ── 30-day Readmission ────────────────────────────────────
    readmission_result: Dict[str, Any] = {"readmission_probability": None}
    if ra_row is not None:
        try:
            ra_prob = round(float(bundle["readmission_model"].predict_proba(ra_row)[0][1]), 6)
            ra_class = int(bundle["readmission_model"].predict(ra_row)[0])
            readmission_result = {
                "predicted_class": ra_class,
                "readmission_probability": ra_prob,
                "readmission_percentage": round(ra_prob * 100, 2),
                "risk_level": _risk_label(ra_prob),
            }
        except Exception as exc:
            readmission_result = {"readmission_probability": None, "error": str(exc)}

    # ── SHAP ──────────────────────────────────────────────────
    shap_result = {
        "mortality": _compute_mortality_shap(bundle, row),
        "los": _compute_los_shap(bundle, los_row) if los_row is not None else [],
        "readmission": _compute_readmission_shap(bundle, ra_row) if ra_row is not None else [],
    }

    return {
        "mortality": {
            "predicted_class": predicted_class,
            "death_probability": round(death_probability, 6),
            "death_percentage": round(death_probability * 100, 2),
            "risk_level": _risk_label(death_probability),
            "vote_fraction": round(vote_sum / model_count, 6) if model_count else 0.0,
            "votes": votes,
            "model_probabilities": {k: round(v, 6) for k, v in probabilities.items()},
        },
        "length_of_stay": los_result,
        "readmission_30d": readmission_result,
        "shap": shap_result,
        "engineered_features": row.iloc[0].replace({float("nan"): None}).to_dict(),
        "model_version": bundle.get("model_version", "unknown"),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        method = _method(event)
        path = _path(event)

        if method == "OPTIONS":
            return _response(200, {"ok": True})

        if method == "GET" and path == "/health":
            bundle = _load_bundle()
            return _response(
                200,
                {
                    "ok": True,
                    "model_version": bundle.get("model_version", "unknown"),
                    "models": list(bundle.get("models", {}).keys()),
                },
            )

        if method == "POST" and path == "/predict":
            if not _check_api_key(event):
                return _response(401, {"error": "Unauthorized. Missing or invalid x-api-key."})
            payload = _parse_body(event)
            patient = payload.get("patient") if isinstance(payload, dict) else None
            if not patient:
                return _response(400, {"error": "Request JSON must contain a top-level 'patient' object."})
            result = _predict(patient)
            return _response(200, result)

        if method == "POST" and path == "/suggest":
            if not _check_api_key(event):
                return _response(401, {"error": "Unauthorized. Missing or invalid x-api-key."})
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            if not gemini_key:
                return _response(503, {"error": "Gemini API key not configured on the server."})
            payload = _parse_body(event)
            patient = payload.get("patient")
            prediction = payload.get("prediction")
            if not patient or not prediction:
                return _response(400, {"error": "Request must contain 'patient' and 'prediction' objects."})
            suggestions = _call_gemini(patient, prediction, gemini_key)
            return _response(200, {"suggestions": suggestions})

        return _response(404, {"error": "Not found. Use GET /health, POST /predict, or POST /suggest."})

    except json.JSONDecodeError:
        return _response(400, {"error": "Body must be valid JSON."})
    except Exception as exc:  # pragma: no cover
        return _response(500, {"error": str(exc)})
