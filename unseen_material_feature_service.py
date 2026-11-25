import os
import re
import math
import logging
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition
from mendeleev import element

from pymatgen.core import Element

# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "battery_voltage_dataset_3977.csv"

PIPELINE_PATH = (
    BASE_DIR / "CrystalVolt_XGBoost_15_Feature_Pipeline.pkl"
)

MP_API_KEY = os.getenv("MP_API_KEY")

FEATURES = [
    "Working_Ion",
    "Space group number",
    "Boiling temperature",
    "Column number",
    "Is non-metal",
    "Covalent radius",
    "Mendeleev number",
    "Is metal",
    "First ionization energy",
    "Crystal_Lattice_Type",
    "Polarisability",
    "Atomic volume",
    "Atomic weight",
    "No. of valence",
    "Density"
]


# =====================================================
# 1. PARSE ELECTRODE FORMULA
# =====================================================

def parse_electrode_formula(electrode_formula):

    if not electrode_formula:
        raise ValueError("Electrode formula cannot be empty.")

    formula = str(electrode_formula).strip().replace(" ", "")

    # Example: Li0-2AgAsF6
    range_match = re.match(
        r"^([A-Z][a-z]?)(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(.+)$",
        formula
    )

    if range_match:
        ion, minimum, maximum, host = range_match.groups()

        concentration_min = float(minimum)
        concentration_max = float(maximum)

    else:

        # Example: Li0.2FePO4
        fixed_match = re.match(
            r"^([A-Z][a-z]?)(\d+(?:\.\d+)?)(.+)$",
            formula
        )

        if fixed_match:
            ion, concentration, host = fixed_match.groups()

            concentration_min = float(concentration)
            concentration_max = float(concentration)

        else:

            # Example: LiFePO4
            no_concentration_match = re.match(
                r"^([A-Z][a-z]?)([A-Z].+)$",
                formula
            )

            if not no_concentration_match:
                raise ValueError(
                    f"Invalid electrode formula: {electrode_formula}"
                )

            ion, host = no_concentration_match.groups()

            concentration_min = 1.0
            concentration_max = 1.0

    if concentration_max < concentration_min:
        raise ValueError("Invalid concentration range.")

    Composition(host)

    return (
        ion,
        concentration_min,
        concentration_max,
        host
    )


# =====================================================
# 2. MATERIALS PROJECT CONNECTION
# =====================================================

def get_mp_client():

    if not MP_API_KEY:
        raise RuntimeError(
            "MP_API_KEY is missing from your .env file."
        )

    return MPRester(MP_API_KEY)


def get_reduced_formula(formula):

    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def get_doc_value(doc, field, default=None):

    if isinstance(doc, dict):
        return doc.get(field, default)

    return getattr(doc, field, default)


def get_symmetry_details(doc):

    symmetry = get_doc_value(doc, "symmetry")

    if symmetry is None:
        return None, None

    if isinstance(symmetry, dict):
        crystal_system = symmetry.get("crystal_system")
        space_group_number = symmetry.get("number")
    else:
        crystal_system = getattr(
            symmetry, "crystal_system", None
        )
        space_group_number = getattr(
            symmetry, "number", None
        )

    if space_group_number is not None:
        space_group_number = int(space_group_number)

    if crystal_system is not None:
        crystal_system = str(crystal_system)

    return crystal_system, space_group_number


# =====================================================
# 3. SEARCH HOST STRUCTURES IN MATERIALS PROJECT
# =====================================================

def get_mp_host_options(host_formula):

    requested_formula = get_reduced_formula(host_formula)

    if requested_formula is None:
        raise ValueError("Invalid host formula.")

    fields = [
        "material_id",
        "formula_pretty",
        "composition",
        "density",
        "symmetry",
        "energy_above_hull"
    ]

    try:
        with get_mp_client() as mpr:

            documents = mpr.materials.summary.search(
                formula=host_formula,
                fields=fields
            )

    except Exception as exc:
        logger.exception("Materials Project search failed.")
        raise RuntimeError(
            f"Materials Project search failed: {exc}"
        ) from exc

    options = []

    for doc in documents:

        formula_pretty = get_doc_value(
            doc, "formula_pretty"
        )

        if not formula_pretty:
            continue

        actual_formula = get_reduced_formula(
            formula_pretty
        )

        if actual_formula != requested_formula:
            continue

        material_id = get_doc_value(
            doc, "material_id"
        )

        if material_id is None:
            continue

        crystal_system, space_group_number = (
            get_symmetry_details(doc)
        )

        density = get_doc_value(doc, "density")

        energy = get_doc_value(
            doc, "energy_above_hull"
        )

        options.append({
            "material_id": str(material_id),
            "formula_pretty": str(formula_pretty),
            "crystal_system": crystal_system,
            "space_group_number": space_group_number,
            "density": (
                float(density) if density is not None else None
            ),
            "energy_above_hull": (
                float(energy) if energy is not None else None
            )
        })

    options.sort(
        key=lambda item: (
            item["energy_above_hull"] is None,
            item["energy_above_hull"]
            if item["energy_above_hull"] is not None
            else float("inf")
        )
    )

    return options


# =====================================================
# 4. RETRIEVE USER-SELECTED MATERIAL
# =====================================================

def get_selected_mp_host(host_formula, selected_material_id):

    if not selected_material_id:
        raise ValueError(
            "Please select a Materials Project structure."
        )

    requested_formula = get_reduced_formula(host_formula)

    fields = [
        "material_id",
        "formula_pretty",
        "composition",
        "density",
        "symmetry",
        "energy_above_hull"
    ]

    try:
        with get_mp_client() as mpr:

            documents = mpr.materials.summary.search(
                material_ids=[selected_material_id],
                fields=fields
            )

    except Exception as exc:
        logger.exception("Selected material retrieval failed.")
        raise RuntimeError(
            f"Could not retrieve selected material: {exc}"
        ) from exc

    if not documents:
        raise LookupError(
            f"Material {selected_material_id} was not found."
        )

    selected_doc = documents[0]

    actual_id = str(
        get_doc_value(selected_doc, "material_id")
    )

    if actual_id != str(selected_material_id):
        raise ValueError("Selected material ID mismatch.")

    selected_formula = get_doc_value(
        selected_doc, "formula_pretty"
    )

    actual_formula = get_reduced_formula(selected_formula)

    if actual_formula != requested_formula:
        raise ValueError(
            f"Selected material formula {selected_formula} "
            f"does not match requested host {host_formula}."
        )

    return selected_doc
# =====================================================
# 5. LOAD TRAINING DATASET
# =====================================================

@lru_cache(maxsize=1)
def load_training_data():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


# =====================================================
# 6. ELEMENTAL PROPERTY EXTRACTION
# =====================================================

def safe_property(el, property_name):

    try:
        value = getattr(el, property_name)

        if callable(value):
            value = value()

        if value is None:
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def get_first_ionization_energy(el):

    try:
        ionenergies = el.ionenergies

        if not ionenergies:
            return None

        value = ionenergies.get(1)

        if value is None:
            return None

        return float(value)

    except Exception:
        return None


def get_element_descriptors(symbol):

    try:
        el = element(symbol)
        pm_el = Element(symbol)
    except Exception as exc:
        raise ValueError(
            f"Element data unavailable for {symbol}: {exc}"
        )

    def get_value(mendeleev_name, pymatgen_name=None):

        value = safe_property(el, mendeleev_name)

        if value is None and pymatgen_name:
            try:
                value = pm_el.data.get(pymatgen_name)
                if value is not None:
                    value = float(value)
            except (TypeError, ValueError, AttributeError):
                value = None

        return value

    # Metal / non-metal classification
    try:
        is_metal = bool(pm_el.is_metal)
        is_metalloid = bool(pm_el.is_metalloid)

        metal_value = 1.0 if is_metal else 0.0

        if is_metalloid:
            nonmetal_value = None
        else:
            nonmetal_value = 0.0 if is_metal else 1.0

    except Exception:
        metal_value = None
        nonmetal_value = None

    return {
        "Boiling temperature": get_value(
            "boiling_point", "Boiling point"
        ),

        "Column number": get_value("group_id"),

        "Is non-metal": nonmetal_value,

        "Covalent radius": get_value(
            "covalent_radius_pyykko"
        ),

        "Mendeleev number": get_value(
            "mendeleev_number", "Mendeleev no"
        ),

        "Is metal": metal_value,

        "First ionization energy": get_first_ionization_energy(el),

        "Polarisability": get_value(
            "dipole_polarizability"
        ),

        "Atomic volume": get_value(
            "atomic_volume", "Atomic volume"
        ),

        "Atomic weight": get_value(
            "atomic_weight", "Atomic mass"
        ),

        "No. of valence": get_value("nvalence")
    }

# =====================================================
# 7. CALCULATE WEIGHTED HOST FEATURES
# =====================================================

def calculate_weighted_host_features(host_formula):

    composition = Composition(host_formula)

    element_amounts = composition.get_el_amt_dict()

    total_atoms = sum(element_amounts.values())

    if total_atoms <= 0:
        raise ValueError(
            f"Invalid host composition: {host_formula}"
        )

    feature_names = [
        "Boiling temperature",
        "Column number",
        "Is non-metal",
        "Covalent radius",
        "Mendeleev number",
        "Is metal",
        "First ionization energy",
        "Polarisability",
        "Atomic volume",
        "Atomic weight",
        "No. of valence"
    ]

    weighted_values = {
        feature: 0.0
        for feature in feature_names
    }

    missing_features = set()

    for symbol, amount in element_amounts.items():

        fraction = amount / total_atoms

        descriptors = get_element_descriptors(symbol)

        for feature in feature_names:

            value = descriptors[feature]

            if value is None:
                missing_features.add(feature)
                continue

            weighted_values[feature] += (
                fraction * value
            )

    if missing_features:

        missing = ", ".join(
            sorted(missing_features)
        )

        raise ValueError(
            "Missing elemental properties for "
            f"{host_formula}: {missing}. "
            "No median substitution was applied."
        )

    return weighted_values, []


# =====================================================
# 8. BUILD THE EXACT 15 FEATURES
# =====================================================

def build_15_features(
    electrode_formula,
    selected_material_id
):

    (
        working_ion,
        concentration_min,
        concentration_max,
        host_formula
    ) = parse_electrode_formula(electrode_formula)

    # Retrieve the exact structure selected by the user
    host_doc = get_selected_mp_host(
        host_formula,
        selected_material_id
    )

    crystal_system, space_group_number = (
        get_symmetry_details(host_doc)
    )

    if crystal_system is None:
        raise ValueError(
            "Crystal system unavailable."
        )

    if space_group_number is None:
        raise ValueError(
            "Space group number unavailable."
        )

    density_g_cm3 = get_doc_value(
        host_doc, "density"
    )

    if density_g_cm3 is None:
        raise ValueError(
            "Density unavailable from Materials Project."
        )

    # Convert g/cm3 to kg/m3.
    # Verify that this matches the training CSV convention.
    density = float(density_g_cm3) * 1000.0

    # Extract the 11 composition-weighted descriptors
    weighted_features, warnings = (
        calculate_weighted_host_features(host_formula)
    )

    features = {

        "Working_Ion": working_ion,

        "Space group number": str(
            space_group_number
        ),

        "Boiling temperature": weighted_features[
            "Boiling temperature"
        ],

        "Column number": weighted_features[
            "Column number"
        ],

        "Is non-metal": weighted_features[
            "Is non-metal"
        ],

        "Covalent radius": weighted_features[
            "Covalent radius"
        ],

        "Mendeleev number": weighted_features[
            "Mendeleev number"
        ],

        "Is metal": weighted_features[
            "Is metal"
        ],

        "First ionization energy": weighted_features[
            "First ionization energy"
        ],

        "Crystal_Lattice_Type": str(
            crystal_system
        ),

        "Polarisability": weighted_features[
            "Polarisability"
        ],

        "Atomic volume": weighted_features[
            "Atomic volume"
        ],

        "Atomic weight": weighted_features[
            "Atomic weight"
        ],

        "No. of valence": weighted_features[
            "No. of valence"
        ],

        "Density": density
    }

    # Maintain the exact order used during training
    features = {
        name: features[name]
        for name in FEATURES
    }

    warnings.append(
        f"Input concentration range: "
        f"{concentration_min} to {concentration_max}. "
        "Concentration is not a separate input "
        "to this 15-feature model."
    )

    return features, warnings, host_doc


import sys
import xgboost
import joblib
import os

print("\n========== PIPELINE DIAGNOSTICS ==========")
print("Python executable:", sys.executable)
print("XGBoost version:", xgboost.__version__)
print("XGBoost location:", xgboost.__file__)
print("Joblib location:", joblib.__file__)
print("Pipeline path:", PIPELINE_PATH)
print("Pipeline exists:", os.path.exists(PIPELINE_PATH))

if os.path.exists(PIPELINE_PATH):
    print("Pipeline size:", os.path.getsize(PIPELINE_PATH), "bytes")

print("==========================================\n")

bundle = joblib.load(PIPELINE_PATH)
# =====================================================
# 9. LOAD SAVED XGBOOST PIPELINE
# =====================================================

@lru_cache(maxsize=1)
def load_pipeline():

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Model pipeline not found: {PIPELINE_PATH}"
        )

    bundle = joblib.load(PIPELINE_PATH)

    required_keys = [
        "model",
        "preprocessor",
        "feature_columns"
    ]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(
                f"Pipeline is missing required key: {key}"
            )

    if list(bundle["feature_columns"]) != FEATURES:
        raise ValueError(
            "Feature order mismatch between training "
            "and prediction pipeline."
        )

    return (
        bundle["model"],
        bundle["preprocessor"],
        bundle["feature_columns"]
    )


# =====================================================
# 10. VALIDATE FEATURE VALUES
# =====================================================

def validate_features(features):

    if not isinstance(features, dict):
        raise TypeError(
            "Features must be provided as a dictionary."
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    extra = [
        feature
        for feature in features
        if feature not in FEATURES
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected features: {extra}"
        )

    categorical_features = [
        "Working_Ion",
        "Space group number",
        "Crystal_Lattice_Type"
    ]

    for feature in categorical_features:

        value = features[feature]

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"Invalid categorical feature: {feature}"
            )

    numeric_features = [
        feature
        for feature in FEATURES
        if feature not in categorical_features
    ]

    for feature in numeric_features:

        try:
            value = float(features[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature {feature} must be numeric."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Feature {feature} contains an invalid value."
            )

    return True


# =====================================================
# 11. PREDICT UNSEEN MATERIAL VOLTAGE
# =====================================================

def predict_unseen_material(
    electrode_formula,
    selected_material_id
):

    # Build all 15 features
    features, warnings, host_doc = build_15_features(
        electrode_formula,
        selected_material_id
    )
    print("\n" + "=" * 70)
    print("MATERIAL DEBUG")
    print("Formula:", electrode_formula)
    print("Material ID:", selected_material_id)
    print("\nRAW 15 FEATURES:")
    for feature, value in features.items():
        print(f"{feature}: {value}")
    print("=" * 70)

    # Validate extracted features
    validate_features(features)

    # Load trained model and preprocessor
    model, preprocessor, feature_columns = load_pipeline()

    # Create DataFrame in exact training order
    input_df = pd.DataFrame(
        [[features[name] for name in feature_columns]],
        columns=feature_columns
    )

    # Apply the fitted training preprocessor
    transformed_features = preprocessor.transform(
        input_df
    )

    # Predict voltage
    prediction = model.predict(
        transformed_features
    )

    predicted_voltage = float(prediction[0])

    if not math.isfinite(predicted_voltage):
        raise ValueError(
            "Model returned an invalid voltage prediction."
        )

    return {
        "success": True,
        "electrode_formula": electrode_formula,
        "host_formula": parse_electrode_formula(
            electrode_formula
        )[3],
        "material_id": str(
            get_doc_value(host_doc, "material_id")
        ),
        "host_formula_pretty": get_doc_value(
            host_doc, "formula_pretty"
        ),
        "predicted_voltage": predicted_voltage,
        "features_used": features,
        "warnings": warnings
    }


# =====================================================
# 12. FEATURE EXTRACTION VALIDATION
# =====================================================

def validate_unseen_material_features(
    electrode_formula,
    selected_material_id
):

    print("\n" + "=" * 60)
    print("CRYSTALVOLT - UNSEEN MATERIAL FEATURE VALIDATION")
    print("=" * 60)

    print("\nElectrode formula:", electrode_formula)
    print("Selected MP ID:", selected_material_id)

    features, warnings, host_doc = build_15_features(
        electrode_formula,
        selected_material_id
    )

    validate_features(features)

    print("\nEXTRACTED 15 FEATURES")
    print("-" * 60)

    for index, feature in enumerate(FEATURES, start=1):

        print(
            f"{index:02d}. {feature}: {features[feature]}"
        )

    print("\nFEATURE COUNT:", len(features))

    print("\nWARNINGS")
    print("-" * 60)

    if warnings:
        for warning in warnings:
            print("-", warning)
    else:
        print("No warnings.")

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    return features


# =====================================================
# 13. DIRECT PREDICTION TEST
# =====================================================

if __name__ == "__main__":

    print("CrystalVolt unseen-material service")
    print("Import this module from your FastAPI backend.")