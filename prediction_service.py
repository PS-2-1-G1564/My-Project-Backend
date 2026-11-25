
import numpy as np
import pandas as pd

from model_loader import load_models, load_preprocessor


# Cache models and preprocessor
_models = None
_preprocessor = None


def get_ml_components():

    global _models, _preprocessor

    if _models is None:
        _models = load_models()

    if _preprocessor is None:
        _preprocessor = load_preprocessor()

    expected_models = [
        "Random Forest",
        "KRR",
        "SVR",
        "XGBoost",
        "DNN"
    ]

    missing_models = [
        name for name in expected_models
        if name not in _models
    ]

    if missing_models:
        raise RuntimeError(
            f"Models not loaded: {missing_models}"
        )

    return _models, _preprocessor


def predict_voltage(input_features: dict):

    # =====================================
    # 1. LOAD MODELS AND PREPROCESSOR
    # =====================================

    models, preprocessor = get_ml_components()

    # =====================================
    # 2. GET EXPECTED FEATURE NAMES
    # =====================================

    if not hasattr(preprocessor, "feature_names_in_"):
        raise RuntimeError(
            "Preprocessor does not contain feature_names_in_."
        )

    expected_features = list(
        preprocessor.feature_names_in_
    )

    # =====================================
    # 3. VALIDATE INPUT FEATURES
    # =====================================

    missing_features = [
        feature for feature in expected_features
        if feature not in input_features
    ]

    extra_features = [
        feature for feature in input_features
        if feature not in expected_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected input features: {extra_features}"
        )

    # =====================================
    # 4. CREATE INPUT DATAFRAME
    # =====================================

    input_df = pd.DataFrame(
        [[input_features[name] for name in expected_features]],
        columns=expected_features
    )

    print("\n========== INPUT VERIFICATION ==========")
    print("Raw input shape:", input_df.shape)

    # =====================================
    # 5. APPLY PREPROCESSOR
    # =====================================

    processed_features = preprocessor.transform(
        input_df
    )

    if hasattr(processed_features, "toarray"):
        processed_features = processed_features.toarray()

    processed_features = np.asarray(
        processed_features
    )

    print(
        "Processed input shape:",
        processed_features.shape
    )

    if processed_features.shape[1] != 194:
        raise ValueError(
            f"Expected 194 processed features, "
            f"received {processed_features.shape[1]}."
        )

    # =====================================
    # 6. GENERATE ALL FIVE PREDICTIONS
    # =====================================

    predictions = {}

    print("\n========== BACKEND MODEL PREDICTIONS ==========")

    for model_name, model in models.items():

        if model_name == "DNN":

            result = model.predict(
                processed_features,
                verbose=0
            )

            print("\n========== DNN DEBUG ==========")
            print("DNN model object:", model)
            print("DNN input shape:", model.input_shape)
            print("DNN output shape:", model.output_shape)
            print(
                "DNN raw prediction:",
                np.asarray(result)
            )

        else:
            result = model.predict(
                processed_features
            )

        voltage = float(
            np.asarray(result).reshape(-1)[0]
        )

        predictions[model_name] = round(
            voltage,
            4
        )

        print(f"{model_name}: {voltage:.4f} V")

    # =====================================
    # 7. RETURN RESULTS
    # =====================================

    return {
        "status": "success",
        "input_feature_count": len(expected_features),
        "processed_feature_count": processed_features.shape[1],
        "predictions": predictions,
        "unit": "V"
    }
