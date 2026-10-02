"""Prediction and clinical risk scoring module for DiabetesCare AI.

Provides input validation against physiological ranges, model loading,
preprocessing, probability scoring, and risk band classification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd

from ml.preprocess import (
    FEATURE_COLUMNS,
    load_medians,
    prepare_single_input,
)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
DEFAULT_MEDIANS_PATH = os.path.join(ARTIFACTS_DIR, "medians.json")

# Clinical validation ranges defined in Master Context (Section 10)
VALIDATION_RANGES: Dict[str, Dict[str, Union[float, str]]] = {
    "Pregnancies": {
        "min": 0,
        "max": 20,
        "type": "int",
        "label": "Pregnancies count",
    },
    "Glucose": {
        "min": 40.0,
        "max": 300.0,
        "type": "float",
        "label": "Plasma Glucose (mg/dL)",
    },
    "BloodPressure": {
        "min": 30.0,
        "max": 200.0,
        "type": "float",
        "label": "Diastolic Blood Pressure (mm Hg)",
    },
    "SkinThickness": {
        "min": 0.0,
        "max": 100.0,
        "type": "float",
        "label": "Triceps Skinfold Thickness (mm)",
    },
    "Insulin": {
        "min": 0.0,
        "max": 900.0,
        "type": "float",
        "label": "2-Hour Serum Insulin (μU/mL)",
    },
    "BMI": {
        "min": 10.0,
        "max": 70.0,
        "type": "float",
        "label": "Body Mass Index (BMI)",
    },
    "DiabetesPedigreeFunction": {
        "min": 0.05,
        "max": 2.50,
        "type": "float",
        "label": "Diabetes Pedigree Function",
    },
    "Age": {
        "min": 1,
        "max": 120,
        "type": "int",
        "label": "Age in years",
    },
}


def validate_clinical_inputs(
    inputs: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validates user clinical inputs against medically plausible ranges.

    Args:
        inputs: Dictionary of input feature values.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    for feature, rule in VALIDATION_RANGES.items():
        val = inputs.get(feature)
        if val is None or str(val).strip() == "":
            errors.append(f"Missing required field: {rule['label']}.")
            continue

        try:
            num_val = float(val)
        except (ValueError, TypeError):
            errors.append(
                f"{rule['label']} must be a valid number (received: '{val}')."
            )
            continue

        if rule["type"] == "int" and not num_val.is_integer():
            errors.append(f"{rule['label']} must be a whole number.")

        if num_val < rule["min"] or num_val > rule["max"]:
            errors.append(
                f"{rule['label']} must be between {rule['min']} and {rule['max']} (received: {num_val})."
            )

    return (len(errors) == 0, errors)


def determine_risk_band(probability: float) -> Tuple[str, str, str]:
    """Categorizes predicted diabetes probability into standardized risk bands.

    Thresholds defined in Master Context Section 6:
    - Low: < 0.30
    - Moderate: 0.30 to 0.60
    - High: > 0.60

    Args:
        probability: Float between 0.0 and 1.0.

    Returns:
        Tuple of (risk_band_name, hex_color, badge_style_class).
    """
    if probability < 0.30:
        return "Low", "#10b981", "badge-low"
    elif probability <= 0.60:
        return "Moderate", "#f59e0b", "badge-moderate"
    else:
        return "High", "#ef4444", "badge-high"


def get_lifestyle_recommendations(risk_band: str) -> List[str]:
    """Provides evidence-based lifestyle recommendations tailored to risk level.

    Args:
        risk_band: 'Low', 'Moderate', or 'High'.

    Returns:
        List of actionable health and lifestyle recommendations.
    """
    if risk_band == "Low":
        return [
            "Maintain an active lifestyle with at least 150 minutes of moderate aerobic exercise weekly.",
            "Continue eating a fiber-rich, balanced diet with whole grains, lean proteins, and leafy greens.",
            "Stay hydrated and avoid sugar-sweetened beverages.",
            "Schedule regular preventive annual checkups and routine health screenings.",
        ]
    elif risk_band == "Moderate":
        return [
            "Consult a physician or registered dietitian for targeted preventative dietary screening.",
            "Monitor fasting blood glucose levels every 3 to 6 months.",
            "Incorporate resistance training 2-3 times per week to boost insulin sensitivity.",
            "Reduce intake of refined carbohydrates, ultra-processed foods, and added sugars.",
            "Target a gradual 5-7% weight reduction if body mass index is elevated.",
        ]
    else:  # High
        return [
            "Promptly consult a healthcare provider or endocrinologist for a diagnostic HbA1c test.",
            "Engage in structured medical nutritional therapy with a certified diabetes specialist.",
            "Maintain a daily log of physical activity, meals, and blood pressure readings.",
            "Avoid skipping meals to prevent blood glucose spikes and reactive hypoglycemia.",
            "Prioritize stress management and 7-9 hours of restorative sleep per night.",
        ]


def load_model_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    medians_path: str = DEFAULT_MEDIANS_PATH,
) -> Tuple[Any, Dict[str, float]]:
    """Loads trained model and imputation medians from disk.

    Args:
        model_path: Path to model.joblib.
        medians_path: Path to medians.json or imputer.joblib.

    Returns:
        Tuple of (model, medians_dict).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Please run `python ml/train.py` first."
        )

    model = joblib.load(model_path)
    medians = load_medians(medians_path)
    return model, medians


def predict_diabetes_risk(
    raw_inputs: Dict[str, Any],
    model: Optional[Any] = None,
    medians: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Runs full inference pipeline on a single patient record.

    Args:
        raw_inputs: Raw clinical inputs dictionary.
        model: Optional pre-loaded model (loads from disk if None).
        medians: Optional pre-loaded medians (loads from disk if None).

    Returns:
        Dictionary containing prediction, probability, risk band, and preprocessed data.
    """
    if model is None or medians is None:
        disk_model, disk_medians = load_model_artifacts()
        model = model or disk_model
        medians = medians or disk_medians

    # Preprocess with identical zero-handling & training-median imputation
    X_processed = prepare_single_input(raw_inputs, medians)

    # Compute probability and binary decision
    probs = model.predict_proba(X_processed)[0]
    probability = float(probs[1])
    prediction = int(1 if probability >= 0.50 else 0)

    risk_band, risk_color, badge_class = determine_risk_band(probability)
    recommendations = get_lifestyle_recommendations(risk_band)

    return {
        "probability": round(probability, 4),
        "prediction": prediction,
        "risk_band": risk_band,
        "risk_color": risk_color,
        "badge_class": badge_class,
        "recommendations": recommendations,
        "processed_df": X_processed,
        "raw_inputs": raw_inputs,
    }
