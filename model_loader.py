
import os
import joblib

from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_models():
    models = {}

    print("\n========== LOADING ML MODELS ==========\n")

    # Load Random Forest, KRR and SVR
    model_files = {
        "Random Forest": "Random_Forest_Model.pkl",
        "KRR": "KRR_Model.pkl",
        "SVR": "SVR_Model.pkl"
    }

    for model_name, filename in model_files.items():

        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        models[model_name] = joblib.load(path)

        print(f"[SUCCESS] {model_name} loaded.")
        print(f"File: {path}")

    # Load XGBoost from native JSON
    xgb_path = os.path.join(
        MODEL_DIR,
        "XGBoost_Model.json"
    )

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(xgb_path)

    xgb_model = XGBRegressor()
    xgb_model.load_model(xgb_path)

    models["XGBoost"] = xgb_model

    print("[SUCCESS] XGBoost loaded from JSON.")
    print(f"File: {xgb_path}")

    # Load DNN
    dnn_path = os.path.join(
        MODEL_DIR,
        "DNN_Model.keras"
    )

    if not os.path.exists(dnn_path):
        raise FileNotFoundError(dnn_path)

    models["DNN"] = load_model(dnn_path)

    print("[SUCCESS] DNN loaded.")
    print(f"DNN file path: {dnn_path}")
    print("DNN input shape:", models["DNN"].input_shape)
    print("DNN output shape:", models["DNN"].output_shape)

    print("\nAll five models loaded successfully.")

    return models


def load_preprocessor():

    print("\n========== LOADING PREPROCESSOR ==========\n")

    path = os.path.join(
        MODEL_DIR,
        "Preprocessor.pkl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    preprocessor = joblib.load(path)

    print("[SUCCESS] Preprocessor loaded.")
    print("Preprocessor file:", path)
    print(
        "Expected raw features:",
        len(preprocessor.feature_names_in_)
    )

    return preprocessor
