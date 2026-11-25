# =====================================================
# CRYSTALVOLT - UNSEEN MATERIAL SHAP SERVICE
# Uses the saved 15-feature XGBoost pipeline
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from unseen_material_feature_service import (
    FEATURES,
    DATASET_PATH,
    build_15_features,
    load_pipeline
)


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


# Cache the SHAP explainer
_unseen_explainer = None


def get_unseen_explainer():
    global _unseen_explainer

    if _unseen_explainer is None:
        model, _, _ = load_pipeline()
        _unseen_explainer = shap.TreeExplainer(model)

    return _unseen_explainer


# =====================================================
# MAP TRANSFORMED FEATURES BACK TO ORIGINAL 15 FEATURES
# =====================================================

def get_feature_groups(preprocessor):

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    groups = {feature: [] for feature in FEATURES}

    # Match the longest original feature name first.
    sorted_features = sorted(FEATURES, key=len, reverse=True)

    for index, transformed_name in enumerate(transformed_names):

        name = str(transformed_name)

        # Remove ColumnTransformer prefix, e.g. cat__ or num__.
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
                "Could not map transformed SHAP feature "
                f"'{transformed_name}' to an original feature."
            )

        groups[matched_feature].append(index)

    empty_groups = [
        feature
        for feature, indexes in groups.items()
        if not indexes
    ]

    if empty_groups:
        raise ValueError(
            f"SHAP feature mapping is incomplete: {empty_groups}"
        )

    return groups


# =====================================================
# AGGREGATE ENCODED SHAP VALUES INTO 15 FEATURES
# =====================================================

def aggregate_shap_values(shap_values, groups):

    aggregated = np.zeros(
        (shap_values.shape[0], len(FEATURES)),
        dtype=float
    )

    for feature_index, feature in enumerate(FEATURES):

        encoded_indexes = groups[feature]

        aggregated[:, feature_index] = np.sum(
            shap_values[:, encoded_indexes],
            axis=1
        )

    return aggregated


# =====================================================
# PREPARE MODEL INPUT
# =====================================================

def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def transform_features(raw_features, preprocessor):

    missing = [name for name in FEATURES if name not in raw_features]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    input_df = pd.DataFrame(
        [[raw_features[name] for name in FEATURES]],
        columns=FEATURES
    )

    return to_dense(preprocessor.transform(input_df))


# =====================================================
# CALCULATE SHAP VALUES
# =====================================================

def calculate_shap_values(transformed):

    explainer = get_unseen_explainer()

    explanation = explainer(transformed)

    values = np.asarray(explanation.values)

    # Regression should return samples x transformed features.
    if values.ndim == 3:
        if values.shape[-1] != 1:
            raise ValueError(
                "Unexpected multi-output SHAP result."
            )
        values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP array shape: {values.shape}"
        )

    return values


# =====================================================
# GLOBAL SHAP DATA FOR THREE FRONTEND CHARTS
# =====================================================

def get_unseen_global_shap(sample_size=200):

    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path,
        dtype={
            "Working_Ion": str
        }
    )

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing features: {missing}"
        )

    # Deterministic sample from the training dataset.
    sample = df[FEATURES].sample(
        n=min(int(sample_size), len(df)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    model, preprocessor, feature_columns = load_pipeline()

    if list(feature_columns) != FEATURES:
        raise ValueError(
            "Pipeline feature order does not match the 15 features."
        )

    transformed = to_dense(preprocessor.transform(sample))

    shap_values = calculate_shap_values(transformed)

    groups = get_feature_groups(preprocessor)

    grouped_values = aggregate_shap_values(
        shap_values,
        groups
    )

    # Mean absolute SHAP importance.
    mean_absolute = np.mean(
        np.abs(grouped_values),
        axis=0
    )

    top_indexes = np.argsort(mean_absolute)[::-1][:10]

    feature_importance = [
        {
            "feature": FEATURES[index],
            "mean_abs_shap": round(
                float(mean_absolute[index]), 6
            )
        }
        for index in top_indexes
    ]

    # SHAP distribution across sampled records.
    # Numeric inputs stay numeric (None if NaN) so the frontend can
    # map low values to blue and high values to pink.
    summary_points = []

    for index in top_indexes:

        feature = FEATURES[index]
        column = sample[feature]
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
                )
            })

    # Dependence data for the most important numeric feature.
    numeric_indexes = [
        index
        for index in top_indexes
        if pd.api.types.is_numeric_dtype(sample[FEATURES[index]])
    ]

    if not numeric_indexes:
        raise ValueError(
            "No numeric feature is available for dependence analysis."
        )

    impact_index = int(numeric_indexes[0])
    impact_feature = FEATURES[impact_index]

    # Drop rows with NaN feature values so the JSON stays valid.
    pairs = [
        (safe_float(value), round(float(shap_value), 6))
        for value, shap_value in zip(
            sample[impact_feature],
            grouped_values[:, impact_index],
        )
    ]
    pairs = [(v, s) for v, s in pairs if v is not None]

    impact = {
        "feature_name": impact_feature,
        "feature_values": [v for v, _ in pairs],
        "shap_values": [s for _, s in pairs]
    }

    return {
        "status": "success",
        "model": "XGBoost",
        "feature_count": len(FEATURES),
        "sample_size": len(sample),
        "feature_importance": feature_importance,
        "summary": {
            "feature_names": [
                FEATURES[index]
                for index in top_indexes
            ],
            "points": summary_points
        },
        "impact": impact,
        "note": (
            "SHAP values are calculated using the saved "
            "15-feature XGBoost pipeline. Encoded SHAP "
            "contributions are aggregated to their original "
            "input features. Global results use a deterministic "
            "sample from the supplied dataset."
        )
    }


# =====================================================
# LOCAL SHAP FOR THE SELECTED UNSEEN MATERIAL
# =====================================================

def explain_unseen_material(
    electrode_formula,
    selected_material_id
):

    features, warnings, _host_doc = build_15_features(
        electrode_formula,
        selected_material_id
    )

    model, preprocessor, feature_columns = load_pipeline()

    if list(feature_columns) != FEATURES:
        raise ValueError(
            "Pipeline feature order mismatch."
        )

    transformed = transform_features(
        features,
        preprocessor
    )
    print("\n========== LOCAL SHAP BACKEND DEBUG ==========")
    print("Selected Material ID:", selected_material_id)
    print("Electrode Formula:", electrode_formula)

    print("\nRAW 15 FEATURES:")
    for name, value in features.items():
        print(f"{name}: {value}")

    print("\nTRANSFORMED MODEL INPUT:")
    print(transformed)

    print("\n============================================")

    shap_values = calculate_shap_values(transformed)

    groups = get_feature_groups(preprocessor)

    grouped_values = aggregate_shap_values(
        shap_values,
        groups
    )[0]
    print("\nLOCAL SHAP CONTRIBUTIONS:")
    for feature, value in zip(FEATURES, grouped_values):
        print(f"{feature}: {value}")

    print("SHAP SUM:", np.sum(grouped_values))
    print("MODEL PREDICTION:", model.predict(transformed))
    print("============================================\n")

    prediction = float(
        np.asarray(model.predict(transformed)).reshape(-1)[0]
    )

    if not np.isfinite(prediction):
        raise ValueError(
            "Model returned an invalid prediction."
        )

    base_value = float(
        np.asarray(get_unseen_explainer().expected_value).reshape(-1)[0]
    )

    sorted_indexes = np.argsort(np.abs(grouped_values))[::-1]

    top_features = [
        {
            "feature": FEATURES[index],
            "property_name": FEATURES[index],
            "shap_value": round(float(grouped_values[index]), 6),
            "feature_value": clean_value(features[FEATURES[index]])
        }
        for index in sorted_indexes
    ]

    return {
        "status": "success",
        "explanation_type": "local",
        "model": "XGBoost",
        "electrode_formula": electrode_formula,
        "material_id": str(selected_material_id),
        "prediction": round(prediction, 6),
        "base_value": round(base_value, 6),
        "shap_sum": round(float(np.sum(grouped_values)), 6),
        "feature_count": len(FEATURES),
        "top_features": top_features,
        "features_used": {
            key: clean_value(value)
            for key, value in features.items()
        },
        "warnings": warnings,
        "note": (
            "Local SHAP contributions are calculated for "
            "the selected unseen material using the saved "
            "15-feature XGBoost pipeline."
        )
    }
