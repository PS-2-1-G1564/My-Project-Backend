
import os
from dotenv import load_dotenv
from mp_api.client import MPRester
import json
import time
import hashlib
import threading
from functools import lru_cache
from pathlib import Path
load_dotenv()

API_KEY = os.getenv("MP_API_KEY")

# --------------------------------------------------
# 1. CONNECT TO MATERIALS PROJECT
# --------------------------------------------------

def get_materials_project_client():
    if not API_KEY:
        raise ValueError("Materials Project API key not found in .env")

    return MPRester(API_KEY)

# --------------------------------------------------
# 2. SEARCH MATERIALS USING CHEMICAL FORMULA
# --------------------------------------------------

def search_materials(formula):

    fields = [
        "material_id",
        "formula_pretty",
        "elements",
        "symmetry",
        "is_stable",
        "energy_above_hull"
    ]

    with get_materials_project_client() as mpr:

        results = mpr.materials.summary.search(
            formula=formula,
            fields=fields
        )

        materials = []

        for material in results:

            symmetry = material.symmetry

            materials.append({

                "material_id": str(material.material_id),

                "formula": material.formula_pretty,

                "elements": [
                    str(element)
                    for element in material.elements
                ],

                "crystal_system": (
                    str(symmetry.crystal_system)
                    if symmetry else None
                ),

                "space_group": (
                    symmetry.symbol
                    if symmetry else None
                ),

                "is_stable": material.is_stable,

                "energy_above_hull_eV_per_atom": (
                    float(material.energy_above_hull)
                    if material.energy_above_hull is not None
                    else None
                )

            })

        return materials

# --------------------------------------------------
# 3. GET COMPLETE INFORMATION USING MATERIAL ID
# --------------------------------------------------

def _fetch_material_by_id(material_id):

    fields = [
        "material_id",
        "formula_pretty",
        "elements",
        "composition",
        "density",
        "volume",
        "nsites",
        "symmetry",
        "structure",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable"
    ]

    with get_materials_project_client() as mpr:

        results = mpr.materials.summary.search(
            material_ids=[material_id],
            fields=fields
        )

        if not results:
            return None

        material = results[0]

        structure = material.structure
        lattice = structure.lattice if structure else None
        symmetry = material.symmetry

        # --------------------------------------------------
        # 4. ATOMIC STRUCTURE
        # --------------------------------------------------

        atomic_species = []
        atomic_positions = []

        if structure:

            for site in structure:

                species = site.species_string

                if species not in atomic_species:
                    atomic_species.append(species)

                atomic_positions.append({

                    "element": species,

                    "fractional_coordinates": [
                        float(x)
                        for x in site.frac_coords
                    ],

                    "cartesian_coordinates": [
                        float(x)
                        for x in site.coords
                    ]

                })

        # --------------------------------------------------
        # 5. COMPOSITION
        # --------------------------------------------------

        composition = {
            str(element): float(amount)
            for element, amount in material.composition.as_dict().items()
        }

        # --------------------------------------------------
        # 6. IDENTITY
        # --------------------------------------------------

        identity = {

            "material_id": str(material.material_id),

            "formula": material.formula_pretty,

            "elements": [
                str(element)
                for element in material.elements
            ],

            "composition": composition

        }

        # --------------------------------------------------
        # 7. PHYSICAL PROPERTIES
        # --------------------------------------------------

        physical_properties = {

            "density_g_per_cm3": material.density,

            "volume_angstrom3": material.volume,

            "number_of_atomic_sites": material.nsites

        }

        # --------------------------------------------------
        # 8. CRYSTAL STRUCTURE
        # --------------------------------------------------

        crystal_structure = {

            "crystal_system": (
                str(symmetry.crystal_system)
                if symmetry else None
            ),

            "space_group": (
                symmetry.symbol
                if symmetry else None
            ),

            "space_group_number": (
                symmetry.number
                if symmetry else None
            ),

            "lattice_parameters": {

                "a_angstrom": lattice.a if lattice else None,

                "b_angstrom": lattice.b if lattice else None,

                "c_angstrom": lattice.c if lattice else None,

                "alpha_degrees": lattice.alpha if lattice else None,

                "beta_degrees": lattice.beta if lattice else None,

                "gamma_degrees": lattice.gamma if lattice else None

            }

        }

        # --------------------------------------------------
        # 9. ATOMIC STRUCTURE
        # --------------------------------------------------

        atomic_structure = {

            "atomic_species": atomic_species,

            "number_of_atoms": len(atomic_positions),

            "atomic_positions": atomic_positions

        }

        # --------------------------------------------------
        # 10. ELECTRONIC PROPERTIES
        # --------------------------------------------------

        electronic_properties = {

            "band_gap_eV": material.band_gap,

            "is_metal": (
                material.band_gap == 0
                if material.band_gap is not None
                else None
            ),

            "source": "Materials Project"

        }

        # --------------------------------------------------
        # 11. STABILITY PROPERTIES
        # --------------------------------------------------

        stability_properties = {

            "formation_energy_per_atom_eV":
                material.formation_energy_per_atom,

            "energy_above_hull_eV_per_atom":
                material.energy_above_hull,

            "is_stable": material.is_stable,

            "source": "Materials Project"

        }

        # --------------------------------------------------
        # 12. BATTERY INFORMATION
        # --------------------------------------------------

        battery_information = {

            "battery_applications": None,

            "metal_ion_type": None,

            "average_voltage_V": None,

            "specific_capacity_mAh_g": None,

            "energy_density_Wh_kg": None,

            "status": (
                "Battery information is retrieved separately "
                "through the battery insertion service."
            )

        }

        # --------------------------------------------------
        # 13. AVAILABILITY AND SCARCITY
        # --------------------------------------------------

        availability = {

            "elements": {},

            "status": (
                "Not integrated. Element abundance, reserves "
                "and supply risk require external sources."
            )

        }

        for element in identity["elements"]:

            availability["elements"][element] = {

                "crustal_abundance": None,

                "global_reserves": None,

                "production": None,

                "supply_risk": None,

                "source": None,

                "data_year": None

            }

        # --------------------------------------------------
        # 14. COST INFORMATION
        # --------------------------------------------------

        cost_information = {

            "element_prices": {},

            "estimated_material_cost": None,

            "status": (
                "Not integrated. Requires dated market prices, "
                "purity and a defensible cost calculation."
            )

        }

        for element in identity["elements"]:

            cost_information["element_prices"][element] = {

                "price": None,

                "currency": None,

                "unit": None,

                "source": None,

                "date": None

            }

        # --------------------------------------------------
        # 15. MATERIAL DESCRIPTION
        # --------------------------------------------------

        description = {

            "overview": (
                f"{material.formula_pretty} is a material record "
                f"in the Materials Project database with material "
                f"ID {material.material_id}."
            ),

            "composition": (
                "The recorded chemical composition contains: "
                f"{', '.join(identity['elements'])}."
            ),

            "structure": (
                "The recorded crystal system is "
                f"{crystal_structure['crystal_system']} and "
                "the space group is "
                f"{crystal_structure['space_group']}."
            ),

            "stability": (
                "The Materials Project stability result is "
                f"{material.is_stable}."
            ),

            "limitation": (
                "This is a database-based summary, not a complete "
                "literature review or verified description of "
                "battery performance and applications."
            )

        }

        # --------------------------------------------------
        # 16. FINAL MATERIAL PROFILE
        # --------------------------------------------------

        material_profile = {

            "identity": identity,

            "physical_properties": physical_properties,

            "crystal_structure": crystal_structure,

            "atomic_structure": atomic_structure,

            "electronic_properties": electronic_properties,

            "stability_properties": stability_properties,

            "battery_information": battery_information,

            "availability": availability,

            "cost_information": cost_information,

            "description": description,

            "sources": [

                {
                    "name": "Materials Project",

                    "material_id": str(material.material_id),

                    "url": (
                        "https://materialsproject.org/materials/"
                        f"{material.material_id}"
                    )

                }

            ]

        }

        return material_profile
    # ==========================================
# MATERIALS PROJECT CACHE
# ==========================================

CACHE_TTL_SECONDS = 7 * 24 * 60 * 60

CACHE_DIR = (
    Path(__file__).resolve().parent
    / "cache"
    / "materials_project"
)

_CACHE_LOCK = threading.RLock()


def _cache_file_for(material_id):
    safe_id = hashlib.sha256(
        str(material_id).encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{safe_id}.json"


@lru_cache(maxsize=256)
def _get_material_profile_cached(material_id):
    cache_file = _cache_file_for(material_id)

    with _CACHE_LOCK:
        if cache_file.exists():
            try:
                age = time.time() - cache_file.stat().st_mtime

                if age <= CACHE_TTL_SECONDS:
                    with cache_file.open("r", encoding="utf-8") as file:
                        return json.load(file)

            except (OSError, json.JSONDecodeError):
                pass

        profile = _fetch_material_by_id(material_id)

        if profile is None:
            return None

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            temporary_file = cache_file.with_suffix(".tmp")

            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(
                    profile,
                    file,
                    ensure_ascii=False,
                    separators=(",", ":")
                )

            os.replace(temporary_file, cache_file)

        except (OSError, TypeError):
            pass

        return profile


def get_material_by_id(material_id):
    return _get_material_profile_cached(
        str(material_id).strip()
    )