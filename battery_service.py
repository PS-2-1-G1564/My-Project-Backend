
import os
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Element

load_dotenv()

API_KEY = os.getenv("MP_API_KEY")


# --------------------------------------------------
# 1. GET BATTERY INFORMATION
# --------------------------------------------------

def get_battery_information(
    material_id,
    formula,
    working_ion="Li"
):

    if not API_KEY:
        raise ValueError(
            "Materials Project API key is missing"
        )

    battery_records = []

    with MPRester(API_KEY) as mpr:

        # Search insertion electrode records
        # using the selected working ion.

        results = mpr.materials.insertion_electrodes.search(
            working_ion=Element(working_ion),
            all_fields=True
        )

        for record in results:

            charge_id = (
                str(record.id_charge)
                if record.id_charge
                else None
            )

            discharge_id = (
                str(record.id_discharge)
                if record.id_discharge
                else None
            )

            # Keep only records related to selected material.

            if (
                charge_id != material_id
                and discharge_id != material_id
            ):
                continue

            selected_role = (
                "charge"
                if charge_id == material_id
                else "discharge"
            )

            battery_records.append({

                "battery_formula": record.battery_formula,

                "battery_type": record.battery_type,

                "working_ion": str(record.working_ion),

                "framework_formula": record.framework_formula,

                "charge_formula": record.formula_charge,

                "discharge_formula": record.formula_discharge,

                "charge_material_id": charge_id,

                "discharge_material_id": discharge_id,

                "selected_material_role": selected_role,

                "average_voltage_V": record.average_voltage,

                "gravimetric_capacity_mAh_g": record.capacity_grav,

                "volumetric_capacity_mAh_cm3": record.capacity_vol,

                "gravimetric_energy_Wh_kg": record.energy_grav,

                "volumetric_energy_Wh_L": record.energy_vol,

                "max_volume_change": record.max_delta_volume,

                "charge_stability": record.stability_charge,

                "discharge_stability": record.stability_discharge,

                "charge_ion_fraction": record.fracA_charge,

                "discharge_ion_fraction": record.fracA_discharge,

                "number_of_steps": record.num_steps,

                "maximum_voltage_step_V": record.max_voltage_step,

                "chemical_system": record.chemsys,

                "elements": [
                    str(element)
                    for element in (record.elements or [])
                ],

                "number_of_elements": record.nelements,

                "voltage_steps": [

                    {

                        "charge_formula": pair.formula_charge,

                        "discharge_formula": pair.formula_discharge,

                        "average_voltage_V": pair.average_voltage,

                        "gravimetric_capacity_mAh_g": pair.capacity_grav,

                        "volumetric_capacity_mAh_cm3": pair.capacity_vol,

                        "gravimetric_energy_Wh_kg": pair.energy_grav,

                        "volumetric_energy_Wh_L": pair.energy_vol,

                        "volume_change": pair.max_delta_volume

                    }

                    for pair in (record.adj_pairs or [])

                ]

            })

    # --------------------------------------------------
    # 2. HANDLE NO MATCHING BATTERY RECORDS
    # --------------------------------------------------

    if not battery_records:

        return {

            "material_id": material_id,

            "formula": formula,

            "working_ion": working_ion,

            "total_matching_records": 0,

            "message": (
                "No matching battery records found "
                "for this material and working ion."
            ),

            "battery_records": []

        }

    # --------------------------------------------------
    # 3. RETURN BATTERY INFORMATION
    # --------------------------------------------------

    return {

        "material_id": material_id,

        "formula": formula,

        "working_ion": working_ion,

        "total_matching_records": len(battery_records),

        "battery_records": battery_records

    }


# --------------------------------------------------
# 4. GET MATERIAL IDs WITH BATTERY RECORDS
# --------------------------------------------------

def get_material_ids_with_battery_records(
    formula,
    working_ion="Li"
):
    """
    Retrieve Material IDs associated with insertion-electrode records.

    Formula is retained for compatibility with existing main.py.
    This implementation searches by working ion.
    """

    if not API_KEY:
        raise ValueError(
            "Materials Project API key is missing"
        )

    matching_material_ids = set()

    with MPRester(API_KEY) as mpr:

        results = mpr.materials.insertion_electrodes.search(
            working_ion=Element(working_ion),
            all_fields=True
        )

        for record in results:

            if record.id_charge:
                matching_material_ids.add(
                    str(record.id_charge)
                )

            if record.id_discharge:
                matching_material_ids.add(
                    str(record.id_discharge)
                )

    return matching_material_ids
