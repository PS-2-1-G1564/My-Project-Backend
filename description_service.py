
"""
description_service.py

Generates 11 sections of a material description using:
1. Materials Project material properties
2. Battery insertion information
3. Crossref research-paper metadata

Missing scientific information is not invented.
"""

import re


def get_value(data, *keys, default=None):
    """Safely retrieve nested dictionary values."""

    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def format_value(value, unit=""):
    """Format numerical values and units."""

    if value is None:
        return "Not available"

    if isinstance(value, float):
        return f"{value:.4f} {unit}".strip()

    return f"{value} {unit}".strip()


# --------------------------------------------------
# CRYSTAL LATTICE DESCRIPTION
# --------------------------------------------------

def describe_lattice_parameters(parameters):

    if not isinstance(parameters, dict) or not parameters:
        return "Lattice parameter information is not available."

    parts = []

    for key, value in parameters.items():

        if value is None:
            continue

        key_lower = key.lower()

        if (
            key_lower.endswith("_degrees")
            or key_lower in {"alpha", "beta", "gamma"}
        ):
            unit = "°"

        elif (
            key_lower.endswith("_angstrom")
            or key_lower in {"a", "b", "c"}
        ):
            unit = "Å"

        else:
            unit = ""

        parts.append(
            f"{key} = {format_value(value, unit)}"
        )

    if not parts:
        return "Lattice parameter information is not available."

    return (
        "The reported lattice parameters are "
        + ", ".join(parts)
        + "."
    )


# --------------------------------------------------
# LITERATURE RELEVANCE FILTER
# --------------------------------------------------

def get_relevant_papers(literature):

    papers = literature.get(
        "papers",
        literature.get("references", [])
    )

    formula = str(
        literature.get("search_query") or ""
    ).lower()

    formula = re.sub(r"\s+", "", formula)

    if not formula:
        return []

    relevant = []

    for paper in papers:

        title = re.sub(
            r"\s+", "", (paper.get("title") or "").lower()
        )

        abstract = re.sub(
            r"\s+", "", (paper.get("abstract") or "").lower()
        )

        if formula in title or formula in abstract:
            relevant.append(paper)

    return relevant


# --------------------------------------------------
# 1. OVERVIEW
# --------------------------------------------------

def generate_overview(profile):

    identity = profile.get("identity", {})

    return {
        "title": "Material Overview",

        "material_id": identity.get("material_id"),

        "formula": identity.get("formula"),

        "description": (
            f"{identity.get('formula', 'This material')} "
            f"is recorded in the Materials Project database "
            f"under material ID "
            f"{identity.get('material_id', 'unknown')}."
        ),

        "source": "Materials Project"
    }


# --------------------------------------------------
# 2. CHEMICAL COMPOSITION
# --------------------------------------------------

def generate_composition(profile):

    identity = profile.get("identity", {})

    return {
        "title": "Chemical Composition",

        "formula": identity.get("formula"),

        "elements": identity.get("elements", []),

        "atomic_composition": identity.get(
            "composition", {}
        ),

        "description": (
            "The material contains the elements listed above. "
            "The atomic composition represents the composition "
            "reported in the selected Materials Project record."
        )
    }


# --------------------------------------------------
# 3. CRYSTAL STRUCTURE
# --------------------------------------------------

def generate_crystal_structure(profile):

    structure = profile.get("crystal_structure", {})

    lattice = structure.get("lattice_parameters", {})

    return {
        "title": "Crystal Structure",

        "crystal_system": structure.get("crystal_system"),

        "space_group": structure.get("space_group"),

        "space_group_number": structure.get(
            "space_group_number"
        ),

        "lattice_parameters": lattice,

        "lattice_description": describe_lattice_parameters(
            lattice
        ),

        "atomic_structure": profile.get(
            "atomic_structure", {}
        ),

        "source": "Materials Project"
    }


# --------------------------------------------------
# 4. PHYSICAL PROPERTIES
# --------------------------------------------------

def generate_physical_properties(profile):

    physical = profile.get("physical_properties", {})

    return {
        "title": "Physical Properties",

        "density_g_per_cm3": physical.get(
            "density_g_per_cm3"
        ),

        "volume_angstrom3": physical.get(
            "volume_angstrom3"
        ),

        "number_of_atomic_sites": physical.get(
            "number_of_atomic_sites"
        ),

        "description": (
            "These are the physical properties available "
            "in the selected Materials Project record."
        )
    }


# --------------------------------------------------
# 5. ELECTRONIC PROPERTIES
# --------------------------------------------------

def generate_electronic_properties(profile):

    electronic = profile.get(
        "electronic_properties", {}
    )

    return {
        "title": "Electronic Properties",

        "band_gap_eV": electronic.get("band_gap_eV"),

        "is_metal": electronic.get("is_metal"),

        "description": (
            "Electronic properties are reported from the "
            "selected Materials Project record. A missing "
            "value means that the property was not available."
        )
    }


# --------------------------------------------------
# 6. STABILITY
# --------------------------------------------------

def generate_stability(profile):

    stability = profile.get(
        "stability_properties", {}
    )

    return {
        "title": "Stability",

        "is_stable": stability.get("is_stable"),

        "formation_energy_per_atom_eV": stability.get(
            "formation_energy_per_atom_eV"
        ),

        "energy_above_hull_eV_per_atom": stability.get(
            "energy_above_hull_eV_per_atom"
        ),

        "description": (
            "Stability information is based on calculated "
            "stability fields in the Materials Project record. "
            "It should not be interpreted as a direct measurement "
            "of long-term battery-cycle stability."
        )
    }


# --------------------------------------------------
# 7. BATTERY APPLICATIONS
# --------------------------------------------------

def generate_battery_applications(battery_data):

    if not battery_data:
        return {
            "title": "Battery Applications",
            "status": "No battery information available.",
            "records": []
        }

    records = battery_data.get("battery_records", [])

    if not records:
        return {
            "title": "Battery Applications",

            "status": battery_data.get(
                "message",
                "No matching battery insertion record was found."
            ),

            "records": []
        }

    applications = []

    for record in records:

        applications.append({

            "battery_formula": record.get(
                "battery_formula"
            ),

            "working_ion": record.get("working_ion"),

            "average_voltage_V": record.get(
                "average_voltage_V"
            ),

            "gravimetric_capacity_mAh_g": record.get(
                "gravimetric_capacity_mAh_g"
            ),

            "volumetric_capacity_mAh_cm3": record.get(
                "volumetric_capacity_mAh_cm3"
            ),

            "gravimetric_energy_Wh_kg": record.get(
                "gravimetric_energy_Wh_kg"
            ),

            "volumetric_energy_Wh_L": record.get(
                "volumetric_energy_Wh_L"
            ),

            "charge_formula": record.get("charge_formula"),

            "discharge_formula": record.get(
                "discharge_formula"
            ),

            "number_of_steps": record.get("number_of_steps"),

            "selected_material_role": record.get(
                "selected_material_role"
            ),

            "voltage_steps": record.get("voltage_steps", [])

        })

    return {
        "title": "Battery Applications",

        "status": (
            "Battery insertion information retrieved successfully."
        ),

        "total_matching_records": len(applications),

        "records": applications,

        "note": (
            "These values come from the retrieved battery insertion "
            "record. They are not machine-learning predictions."
        )
    }


# --------------------------------------------------
# LITERATURE SENTENCE EXTRACTION
# --------------------------------------------------

def extract_evidence_sentences(papers, keywords):

    evidence = []

    for paper in papers:

        abstract = paper.get("abstract")

        if not abstract:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            abstract.strip()
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_lower = sentence.lower()

            if any(
                keyword in sentence_lower
                for keyword in keywords
            ):

                doi = paper.get("doi")

                evidence.append({

                    "statement": sentence,

                    "paper_title": paper.get("title"),

                    "year": paper.get("year"),

                    "doi": doi,

                    "doi_url": (
                        f"https://doi.org/{doi}"
                        if doi else paper.get("url")
                    ),

                    "evidence_source": "Crossref abstract",

                    "scientific_claims_verified": False,

                    "note": (
                        "Extracted from an abstract. Check the "
                        "original paper before treating this as "
                        "a verified material-specific claim."
                    )

                })

    return evidence


# --------------------------------------------------
# 8. WORKING MECHANISM
# --------------------------------------------------

def generate_working_mechanism(literature):

    papers = get_relevant_papers(literature)

    keywords = [
        "intercalation",
        "deintercalation",
        "insertion mechanism",
        "extraction mechanism",
        "ion diffusion",
        "lithium diffusion",
        "sodium diffusion",
        "redox reaction",
        "electron transfer",
        "two-phase reaction"
    ]

    evidence = extract_evidence_sentences(
        papers, keywords
    )

    return {
        "title": "Working Mechanism",

        "status": (
            "evidence_found"
            if evidence else "insufficient_evidence"
        ),

        "description": evidence,

        "message": (
            "Statements extracted from exact-formula search matches."
            if evidence else
            "No qualifying mechanism statements were found."
        )
    }


# --------------------------------------------------
# 9. ADVANTAGES
# --------------------------------------------------

def generate_advantages(literature):

    papers = get_relevant_papers(literature)

    keywords = [
        "improved",
        "enhanced",
        "stable performance",
        "high capacity",
        "higher capacity",
        "good cycling",
        "excellent performance",
        "high energy density",
        "improved capacity",
        "better performance"
    ]

    evidence = extract_evidence_sentences(
        papers, keywords
    )

    return {
        "title": "Advantages",

        "status": (
            "evidence_found"
            if evidence else "insufficient_evidence"
        ),

        "description": evidence,

        "message": (
            "Potential statements extracted from exact-formula abstracts. "
            "Check the original papers."
            if evidence else
            "No advantage statements were found in the available abstracts."
        )
    }


# --------------------------------------------------
# 10. LIMITATIONS
# --------------------------------------------------

def generate_limitations(literature):

    papers = get_relevant_papers(literature)

    keywords = [
        "limitation",
        "poor stability",
        "capacity fading",
        "capacity loss",
        "low conductivity",
        "low diffusion",
        "degradation",
        "voltage decay",
        "poor cycling",
        "volume expansion",
        "short cycle life"
    ]

    evidence = extract_evidence_sentences(
        papers, keywords
    )

    return {
        "title": "Limitations",

        "status": (
            "evidence_found"
            if evidence else "insufficient_evidence"
        ),

        "description": evidence,

        "message": (
            "Potential limitation statements extracted from "
            "exact-formula abstracts. Check the original papers."
            if evidence else
            "No limitation statements were found in the available abstracts."
        )
    }


# --------------------------------------------------
# 11. RESEARCH REFERENCES
# --------------------------------------------------

def generate_references(literature):

    references = get_relevant_papers(literature)

    return {
        "title": "Research References",

        "status": (
            "Exact-formula search matches retrieved. "
            "Verify the original papers."
            if references else
            "No exact-formula references were found by title/abstract screening."
        ),

        "papers": references,

        "related_background": literature.get(
            "related_background", []
        ),

        "note": (
            "Crossref search metadata does not establish that a paper "
            "supports a scientific claim about this material."
        )
    }


# --------------------------------------------------
# FINAL: GENERATE ALL 11 SECTIONS
# --------------------------------------------------

def generate_material_description(
    material_profile,
    battery_data=None,
    literature=None
):

    if not isinstance(material_profile, dict):
        raise TypeError(
            "material_profile must be a dictionary."
        )

    literature = literature or {}

    return {

        "report_type": "Detailed Material Description",

        "material_id": get_value(
            material_profile,
            "identity",
            "material_id"
        ),

        "sections": {

            "1_overview": generate_overview(
                material_profile
            ),

            "2_chemical_composition": generate_composition(
                material_profile
            ),

            "3_crystal_structure": generate_crystal_structure(
                material_profile
            ),

            "4_physical_properties": generate_physical_properties(
                material_profile
            ),

            "5_electronic_properties": generate_electronic_properties(
                material_profile
            ),

            "6_stability": generate_stability(
                material_profile
            ),

            "7_battery_applications": generate_battery_applications(
                battery_data
            ),

            "8_working_mechanism": generate_working_mechanism(
                literature
            ),

            "9_advantages": generate_advantages(
                literature
            ),

            "10_limitations": generate_limitations(
                literature
            ),

            "11_research_references": generate_references(
                literature
            )

        },

        "data_sources": {

            "material_properties": "Materials Project",

            "battery_information": (
                "Materials Project Insertion Electrodes"
                if battery_data else None
            ),

            "literature": (
                "Crossref metadata; exact-formula screened"
                if literature.get("references") else None
            )

        }

    }
