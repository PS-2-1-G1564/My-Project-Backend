
from pathlib import Path
from functools import lru_cache
import pandas as pd


# --------------------------------------------------
# DATASET LOCATION
# --------------------------------------------------

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "battery_voltage_dataset_3977.csv"
)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

@lru_cache(maxsize=1)
def _load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Battery voltage dataset was not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        dtype={
            "Material_ID": str,
            "Electrode_Formula": str
        }
    )

    required = {
        "Material_ID",
        "Electrode_Formula",
        "Working_Ion"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Normalize surrounding whitespace only.
    # Do not rewrite chemical formulas.

    for column in (
        "Material_ID",
        "Electrode_Formula",
        "Working_Ion"
    ):
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    return df


# --------------------------------------------------
# 2. CONVERT VALUES INTO JSON-SAFE VALUES
# --------------------------------------------------

def _json_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


# --------------------------------------------------
# 3. CONVERT DATAFRAME RECORDS INTO JSON
# --------------------------------------------------

def _records_to_json(frame):

    return [
        {
            column: _json_value(value)
            for column, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


# --------------------------------------------------
# 4. SEARCH DATASET BY FORMULA AND WORKING ION
# --------------------------------------------------

def search_dataset_by_formula(
    formula,
    working_ion="Li"
):
    """
    Search the local dataset using exact Electrode_Formula
    and selected Working_Ion.

    Formula existence and working-ion matching are checked
    separately.
    """

    df = _load_dataset()

    formula_key = str(formula).strip()
    ion_key = str(working_ion).strip()

    # Search exact formula

    formula_rows = df[
        df["Electrode_Formula"]
        .str.casefold()
        == formula_key.casefold()
    ].copy()

    # Filter by working ion

    ion_rows = formula_rows[
        formula_rows["Working_Ion"]
        .str.casefold()
        == ion_key.casefold()
    ].copy()

    records = _records_to_json(ion_rows)

    materials = []

    # Group by Material_ID.
    # Different IDs are never merged.

    for material_id, group in ion_rows.groupby(
        "Material_ID",
        dropna=False,
        sort=False
    ):

        if pd.isna(material_id):
            continue

        group_records = _records_to_json(group)

        first = group.iloc[0]

        materials.append({

            "material_id": str(material_id),

            "formula": str(
                first["Electrode_Formula"]
            ),

            "crystal_system": (
                None
                if pd.isna(first.get("Crystal_Lattice_Type"))
                else str(first.get("Crystal_Lattice_Type"))
            ),

            "space_group": (
                None
                if pd.isna(first.get("Space group number"))
                else str(first.get("Space group number"))
            ),

            "working_ion": ion_key,

            "dataset_record_count": len(group_records),

            "dataset_properties": group_records

        })

    return {

        "formula": formula_key,

        "working_ion": ion_key,

        "formula_exists_in_dataset": (
            not formula_rows.empty
        ),

        "matching_record_count": len(records),

        "total_formula_record_count": len(formula_rows),

        "dataset_records": records,

        "materials": materials

    }


# --------------------------------------------------
# 5. GET DATASET RECORDS USING MATERIAL ID
# --------------------------------------------------

def get_dataset_records_by_material_id(material_id):
    """
    Return all dataset rows for a Materials Project ID.
    """

    df = _load_dataset()

    key = str(material_id).strip()

    matches = df[
        df["Material_ID"]
        .str.casefold()
        == key.casefold()
    ].copy()

    return _records_to_json(matches)


# --------------------------------------------------
# 6. DATASET STATUS
# --------------------------------------------------

def get_dataset_status():
    """
    Return dataset information for diagnostics.
    """

    df = _load_dataset()

    return {

        "dataset_file": DATASET_PATH.name,

        "total_records": int(len(df)),

        "unique_material_ids": int(
            df["Material_ID"].nunique()
        ),

        "unique_electrode_formulas": int(
            df["Electrode_Formula"].nunique()
        ),

        "columns": list(df.columns)

    }
