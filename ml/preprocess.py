"""Data preprocessing and imputation module for DiabetesCare AI.

Shared across model training and inference pipelines to ensure consistent
data handling and prevent data leakage.
"""

from typing import Dict, List, Union
import json
import os
import joblib
import numpy as np
import pandas as pd

# Canonical feature ordering expected by the model
FEATURE_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN: str = "Outcome"

# Columns where a measured value of 0 is physiologically invalid (represents missing data)
ZERO_AS_MISSING_COLUMNS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def replace_invalid_zeros(
    df: pd.DataFrame, columns: List[str] = ZERO_AS_MISSING_COLUMNS
) -> pd.DataFrame:
    """Replaces values of 0 with np.nan for specified physiological columns.

    Args:
        df: Input DataFrame.
        columns: List of column names where 0 should be treated as NaN.

    Returns:
        pd.DataFrame with 0s replaced by np.nan in target columns.
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(0, np.nan)
    return df_copy


def compute_training_medians(
    X_train: pd.DataFrame, columns: List[str] = ZERO_AS_MISSING_COLUMNS
) -> Dict[str, float]:
    """Computes median values from the training split for missing-value imputation.

    Args:
        X_train: Training features DataFrame (with 0s already converted to NaN).
        columns: Columns to compute medians for.

    Returns:
        Dictionary mapping column names to training median values.
    """
    medians: Dict[str, float] = {}
    for col in columns:
        if col in X_train.columns:
            median_val = float(X_train[col].median(skipna=True))
            medians[col] = round(median_val, 2)
    return medians


def impute_missing_values(
    df: pd.DataFrame, medians: Dict[str, float]
) -> pd.DataFrame:
    """Imputes NaN values in specified columns using precomputed training medians.

    Args:
        df: DataFrame with NaNs.
        medians: Dictionary of column -> median value.

    Returns:
        Clean DataFrame with missing values filled.
    """
    df_copy = df.copy()
    for col, med_val in medians.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(med_val)
    return df_copy


def save_medians(medians: Dict[str, float], filepath: str) -> None:
    """Saves imputation medians to a JSON or joblib file.

    Args:
        medians: Dictionary of column medians.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if filepath.endswith(".json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(medians, f, indent=2)
    else:
        joblib.dump(medians, filepath)


def load_medians(filepath: str) -> Dict[str, float]:
    """Loads imputation medians from a JSON or joblib file.

    Args:
        filepath: Path to the medians file.

    Returns:
        Dictionary of column medians.
    """
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return joblib.load(filepath)


def prepare_single_input(
    raw_inputs: Dict[str, Union[int, float]], medians: Dict[str, float]
) -> pd.DataFrame:
    """Prepares and validates a single patient's input for model prediction.

    Applies the same zero-to-NaN conversion and training-median imputation
    as performed during training.

    Args:
        raw_inputs: Dictionary with keys matching FEATURE_COLUMNS.
        medians: Precomputed training medians.

    Returns:
        1-row pd.DataFrame matching FEATURE_COLUMNS order, ready for inference.
    """
    # Create single-row DataFrame with expected ordering
    row_data = {col: [raw_inputs.get(col, np.nan)] for col in FEATURE_COLUMNS}
    df = pd.DataFrame(row_data)

    # In zero-as-missing columns, if user input 0 or None/NaN, mark as NaN
    for col in ZERO_AS_MISSING_COLUMNS:
        if col in df.columns:
            val = df.at[0, col]
            if val == 0 or pd.isna(val):
                df.at[0, col] = np.nan

    # Fill NaNs with the training medians
    df = impute_missing_values(df, medians)

    # Ensure float types
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)

    return df
