from pathlib import Path

import numpy as np
import pandas as pd
import shap

from prediction_service import get_ml_components
from dataset_prediction_service import DATASET_PATH, NON_MODEL_COLUMNS


# =====================================================
# CONSTANTS
# =====================================================

DTYPES = {
    "Material_ID": str,
    "Electrode_Formula": str,
    "Working_Ion": str,
}

RANDOM_STATE = 42


# =====================================================
# SMALL HELPERS
# =====================================================

def unwrap(value):
    """Convert numpy scalars to plain Python values."""
    return value.item() if isinstance(value, np.generic) else value


def safe_float(value):
    """Return a finite float, or None for NaN/inf/non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def clean_value(value):
    """Plain Python value, with NaN/inf turned into None (JSON-safe)."""
    value = unwrap(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def to_dense(matrix):
    """Convert sparse matrices to a dense numpy array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def top_indices_by_abs(values, n):
    """Indices of the n largest values by absolute magnitude."""
    return np.argsort(np.abs(values))[::-1][:n]


def load_dataset():
    return pd.read_csv(Path(DATASET_PATH), dtype=DTYPES)


# =====================================================
# CACHES
# =====================================================

_explainer = None
_groups_cache = None


def get_shap_explainer():
    global _explainer

    if _explainer is None:
        models, _ = get_ml_components()
        _explainer = shap.TreeExplainer(models["XGBoost"])

    return _explainer


def get_cached_groups(preprocessor):
    """Preprocessor never changes at runtime, so compute groups once."""
    global _groups_cache

    if _groups_cache is None:
        _groups_cache = get_original_feature_groups(preprocessor)

    return _groups_cache


# =====================================================
# NORMALIZE SHAP OUTPUT
# =====================================================

def normalize_shap_values(values):
    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise ValueError(
                f"Unexpected multi-output SHAP shape: {values.shape}"
            )
        values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values, received {values.shape}"
        )

    return values


# =====================================================
# MAP TRANSFORMED FEATURES TO ORIGINAL FEATURES
# =====================================================

def get_original_feature_groups(preprocessor):
    """
    Map every transformed column to the original input feature
    it came from (one-hot columns map back to their source column).
    """

    original_features = list(preprocessor.feature_names_in_)
    transformed_names = list(preprocessor.get_feature_names_out())

    groups = {feature: [] for feature in original_features}

    # Match longest names first to avoid partial matches.
    sorted_features = sorted(original_features, key=len, reverse=True)

    for index, transformed_name in enumerate(transformed_names):

        name = str(transformed_name)

        # Remove ColumnTransformer prefix.
        if "__" in name:
            name = name.split("__", 1)[1]

        matched_feature = next(
            (
                feature
                for feature in sorted_features
                if name == feature or name.startswith(feature + "_")
            ),
            None,
        )

        if matched_feature is None:
            raise ValueError(
                f"Cannot map transformed feature "
                f"'{transformed_name}' to an original feature."
            )

        groups[matched_feature].append(index)

    empty_groups = [f for f, idx in groups.items() if not idx]

    if empty_groups:
        raise ValueError(
            f"Original features without transformed columns: "
            f"{empty_groups}"
        )

    return groups


# =====================================================
# AGGREGATE SHAP VALUES
# =====================================================

def aggregate_original_shap(shap_values, groups, original_features):
    """
    Combine signed SHAP contributions belonging to
    the same original input feature.
    """

    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP array, got {shap_values.shape}"
        )

    expected_count = sum(len(idx) for idx in groups.values())

    if shap_values.shape[1] != expected_count:
        raise ValueError(
            "SHAP columns do not match transformed feature groups."
        )

    aggregated = np.zeros(
        (shap_values.shape[0], len(original_features)),
        dtype=float,
    )

    for feature_index, feature in enumerate(original_features):
        aggregated[:, feature_index] = np.sum(
            shap_values[:, groups[feature]],
            axis=1,
        )

    return aggregated


# =====================================================
# SHAP -> GROUPED PIPELINE (shared)
# =====================================================

def shap_grouped(processed, preprocessor):
    """
    Run explainer on processed data and aggregate into original features.

    Returns (grouped_values, explanation). The explanation is returned
    too because local SHAP needs explanation.base_values.
    """

    explanation = get_shap_explainer()(processed)
    shap_values = normalize_shap_values(explanation.values)

    original_features = list(preprocessor.feature_names_in_)
    groups = get_cached_groups(preprocessor)

    grouped = aggregate_original_shap(
        shap_values,
        groups,
        original_features,
    )

    return grouped, explanation


def compute_global_shap(sample_size):
    """
    Shared by global importance and plot data.

    Returns (raw_features, original_features, grouped_values).
    """

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = load_dataset()

    _, preprocessor = get_ml_components()
    original_features = list(preprocessor.feature_names_in_)

    raw = df.drop(columns=NON_MODEL_COLUMNS, errors="ignore")

    missing = [n for n in original_features if n not in raw.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing model features: {missing}"
        )

    # Deterministic random sample (avoids bias if the CSV is sorted).
    raw = (
        raw[original_features]
        .sample(n=min(int(sample_size), len(raw)), random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    processed = to_dense(preprocessor.transform(raw))

    grouped, _ = shap_grouped(processed, preprocessor)

    return raw, original_features, grouped


# =====================================================
# TRANSFORM INPUT
# =====================================================

def transform_input(input_features):

    _, preprocessor = get_ml_components()

    expected_features = list(preprocessor.feature_names_in_)

    missing = [n for n in expected_features if n not in input_features]
    extra = [n for n in input_features if n not in expected_features]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    input_df = pd.DataFrame(
        [[input_features[name] for name in expected_features]],
        columns=expected_features,
    )

    processed = to_dense(preprocessor.transform(input_df))

    feature_names = list(preprocessor.get_feature_names_out())

    if processed.shape[1] != len(feature_names):
        raise ValueError(
            f"Processed column count ({processed.shape[1]}) does not "
            f"match feature name count ({len(feature_names)})."
        )

    return processed, feature_names


# =====================================================
# GET KNOWN MATERIAL INPUT
# =====================================================

def get_material_input(material_id, working_ion="Li"):

    df = load_dataset()

    rows = df[
        (df["Material_ID"].str.strip() == str(material_id).strip())
        & (
            df["Working_Ion"].str.strip().str.casefold()
            == str(working_ion).strip().casefold()
        )
    ]

    if rows.empty:
        return None

    if len(rows) > 1:
        raise ValueError(
            "Multiple dataset rows found for this material "
            "and working ion."
        )

    row = rows.iloc[0]

    _, preprocessor = get_ml_components()
    expected_features = list(preprocessor.feature_names_in_)

    missing = [name for name in expected_features if name not in row.index]

    if missing:
        raise ValueError(
            f"Dataset row is missing model features: {missing}"
        )

    # Use only the features the model expects, so extra dataset
    # columns can never trigger "Unexpected features".
    input_features = {name: unwrap(row[name]) for name in expected_features}

    return {
        "material_id": str(row["Material_ID"]),
        "formula": str(row["Electrode_Formula"]),
        "working_ion": str(row["Working_Ion"]),
        "input_features": input_features,
    }


# =====================================================
# LOCAL SHAP - KNOWN MATERIAL
# =====================================================

def explain_material(material_id, working_ion="Li"):

    dataset_input = get_material_input(
        material_id=material_id,
        working_ion=working_ion,
    )

    if dataset_input is None:
        raise ValueError(
            "Material ID and working ion were not found "
            "in the local prediction dataset."
        )

    processed, _ = transform_input(dataset_input["input_features"])

    models, preprocessor = get_ml_components()
    original_features = list(preprocessor.feature_names_in_)

    grouped, explanation = shap_grouped(processed, preprocessor)
    grouped_values = grouped[0]

    base_value = float(
        np.asarray(explanation.base_values).reshape(-1)[0]
    )

    prediction = float(
        np.asarray(models["XGBoost"].predict(processed)).reshape(-1)[0]
    )

    top_features = [
        {
            "feature": original_features[index],
            "property_name": original_features[index],
            "feature_type": "original",
            "encoded_category": None,
            "shap_value": round(float(grouped_values[index]), 6),
            "feature_value": clean_value(
                dataset_input["input_features"][original_features[index]]
            ),
        }
        for index in top_indices_by_abs(grouped_values, 20)
    ]

    return {
        "status": "success",
        "explanation_type": "local",
        "material_id": dataset_input["material_id"],
        "formula": dataset_input["formula"],
        "working_ion": dataset_input["working_ion"],
        "model": "XGBoost",
        "prediction": round(prediction, 6),
        "base_value": round(base_value, 6),
        "shap_sum": round(float(np.sum(grouped_values)), 6),
        "feature_count": len(original_features),
        "top_features": top_features,
        "unit": "V",
        "note": (
            "SHAP contributions are aggregated into original "
            "input features. Positive values push the prediction "
            "higher; negative values push it lower."
        ),
    }


# =====================================================
# GLOBAL SHAP - KNOWN MATERIAL DATASET
# =====================================================

def explain_global(sample_size=200):

    raw_features, original_features, grouped_values = compute_global_shap(
        sample_size
    )

    # Mean absolute SHAP AFTER aggregation.
    mean_abs_values = np.mean(np.abs(grouped_values), axis=0)

    features = [
        {
            "feature": original_features[index],
            "mean_abs_shap": round(float(mean_abs_values[index]), 6),
        }
        for index in top_indices_by_abs(mean_abs_values, 20)
    ]

    return {
        "status": "success",
        "explanation_type": "global",
        "model": "XGBoost",
        "sample_size": len(raw_features),
        "feature_count": len(original_features),
        "top_features": features,
        "unit": "V",
        "note": (
            "Global importance is calculated using mean absolute "
            "SHAP values after aggregating transformed columns "
            "into original input features."
        ),
    }


# =====================================================
# SHAP PLOT DATA FOR FRONTEND
# =====================================================

def get_shap_plot_data(sample_size=200):

    raw_features, original_features, grouped_values = compute_global_shap(
        sample_size
    )

    # -------------------------------------------------
    # Global feature importance
    # -------------------------------------------------

    mean_abs_values = np.mean(np.abs(grouped_values), axis=0)
    top_indices = top_indices_by_abs(mean_abs_values, 10)

    feature_importance = [
        {
            "feature": original_features[index],
            "mean_abs_shap": round(float(mean_abs_values[index]), 6),
        }
        for index in top_indices
    ]

    # -------------------------------------------------
    # SHAP summary points
    # -------------------------------------------------
    # Numeric features are sent as float (None if NaN), categorical
    # as str, so the frontend gets consistent, JSON-safe types.

    summary_points = []

    for index in top_indices:

        feature = original_features[index]
        column = raw_features[feature]
        is_numeric = pd.api.types.is_numeric_dtype(column)

        for row_index, raw_value in enumerate(column):

            summary_points.append({
                "feature": feature,
                "shapValue": round(
                    float(grouped_values[row_index, index]), 6
                ),
                "featureValue": (
                    safe_float(raw_value)
                    if is_numeric
                    else str(unwrap(raw_value))
                ),
            })

    # -------------------------------------------------
    # Dependence data (top numeric feature)
    # -------------------------------------------------

    numeric_indices = [
        index
        for index in top_indices
        if pd.api.types.is_numeric_dtype(
            raw_features[original_features[index]]
        )
    ]

    if not numeric_indices:
        raise ValueError(
            "No numeric original feature is available "
            "for dependence analysis."
        )

    best_index = int(numeric_indices[0])
    best_feature = original_features[best_index]

    # Drop rows with NaN feature values so the JSON stays valid.
    pairs = [
        (safe_float(value), round(float(shap_value), 6))
        for value, shap_value in zip(
            raw_features[best_feature],
            grouped_values[:, best_index],
        )
    ]
    pairs = [(v, s) for v, s in pairs if v is not None]

    impact_data = {
        "feature_name": best_feature,
        "feature_values": [v for v, _ in pairs],
        "shap_values": [s for _, s in pairs],
    }

    return {
        "status": "success",
        "model": "XGBoost",
        "sample_size": len(raw_features),
        "feature_count": len(original_features),
        "feature_importance": feature_importance,
        "summary": {
            "feature_names": [original_features[i] for i in top_indices],
            "points": summary_points,
        },
        "impact": impact_data,
        "note": (
            "SHAP explanations are from XGBoost. Transformed "
            "SHAP contributions are aggregated into original "
            "input features. Global importance uses mean absolute "
            "aggregated SHAP values."
        ),
    }
