
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "battery_voltage_dataset_3977.csv"

NON_MODEL_COLUMNS = [
    "Material_ID",
    "Electrode_Formula",
    "Voltage_V",
    "Oxidation states"
]


def get_prediction_inputs_by_formula(formula, working_ion="Li"):

    """
    Retrieve all matching ML input rows using electrode formula
    and working ion.

    Each matching dataset record is processed separately.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        dtype={
            "Material_ID": str,
            "Electrode_Formula": str,
            "Working_Ion": str
        }
    )

    required_columns = {
        "Material_ID",
        "Electrode_Formula",
        "Working_Ion",
        "Voltage_V",
        *NON_MODEL_COLUMNS
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Step 1: Search by electrode formula
    formula_rows = df[
        df["Electrode_Formula"].str.strip().str.casefold()
        == str(formula).strip().casefold()
    ]

    # Step 2: Filter by working ion
    matching_rows = formula_rows[
        formula_rows["Working_Ion"].str.strip().str.casefold()
        == str(working_ion).strip().casefold()
    ]

    if matching_rows.empty:
        return None

    prediction_inputs = []

    # Step 3: Process every matching record separately
    for _, row in matching_rows.iterrows():

        input_features = row.drop(
            labels=NON_MODEL_COLUMNS
        ).to_dict()

        # Step 4: Validate feature count
        if len(input_features) != 44:
            raise ValueError(
                f"Expected 44 input features, "
                f"but found {len(input_features)}."
            )

        # Convert NumPy values into Python values
        input_features = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in input_features.items()
        }

        prediction_inputs.append({
            "material_id": str(row["Material_ID"]),
            "formula": str(row["Electrode_Formula"]),
            "working_ion": str(row["Working_Ion"]),
            "input_features": input_features,
            "input_feature_count": len(input_features)
        })

    return prediction_inputs
