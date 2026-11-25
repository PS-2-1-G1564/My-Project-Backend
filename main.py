from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware


from shap_service import (
    explain_material,
    explain_global,
    get_shap_plot_data
)

from material_service import search_materials, get_material_by_id
from battery_service import get_battery_information
from prediction_service import predict_voltage

from dataset_service import (
    search_dataset_by_formula,
    get_dataset_records_by_material_id,
    get_dataset_status
)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from dataset_prediction_service import get_prediction_inputs_by_formula

from description_service import generate_material_description
from literature_service import build_literature_description
from paper_content_service import get_papers_with_abstracts

import traceback


from unseen_material_feature_service import (
    parse_electrode_formula,
    get_mp_host_options,
    predict_unseen_material
)

from unseen_shap_service import (
    explain_unseen_material,
    get_unseen_global_shap
)






app = FastAPI(
    title="CrystalVolts API",
    description=(
        "Material search, scientific properties, "
        "battery information and ML voltage prediction"
    ),
    version="1.4"
)


# =====================================
# MODEL EVALUATION METRICS
# =====================================

MODEL_EVALUATION_METRICS = {
    "Random Forest": {
        "MAE": 0.6095685513,
        "RMSE": 1.0323270605,
        "R2": 0.6741137973
    },
    "XGBoost": {
        "MAE": 0.6212714200,
        "RMSE": 1.0225237756,
        "R2": 0.6802738337
    },
    "KRR": {
        "MAE": 0.7174274215,
        "RMSE": 1.0759046486,
        "R2": 0.6460198424
    },
    "SVR": {
        "MAE": 0.6579099872,
        "RMSE": 1.1239897084,
        "R2": 0.6136721476
    },
    "DNN": {
        "MAE": 0.8192293235,
        "RMSE": 1.1943656664,
        "R2": 0.5637796002
    }
}


# =====================================================
# UNSEEN MATERIAL PREDICTION
# =====================================================

@app.get("/materials/host-options")
def host_options(formula: str = Query(..., min_length=1)):
    try:
        (
            working_ion,
            concentration_min,
            concentration_max,
            host_formula
        ) = parse_electrode_formula(formula)

        options = get_mp_host_options(host_formula)

        return {
            "status": "success",
            "electrode_formula": formula,
            "working_ion": working_ion,
            "concentration_min": concentration_min,
            "concentration_max": concentration_max,
            "host_formula": host_formula,
            "total_structures": len(options),
            "structures": options
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))

# @app.get("/predict-unseen")
# def predict_unseen(
#     formula: str = Query(..., min_length=1),
#     material_id: str = Query(..., min_length=1)
# ):
#     try:
#         result = predict_unseen_material(
#             electrode_formula=formula,
#             selected_material_id=material_id
#         )

#         return result

#     except LookupError as error:
#         raise HTTPException(status_code=404, detail=str(error))

#     except ValueError as error:
#         raise HTTPException(status_code=400, detail=str(error))

#     except FileNotFoundError as error:
#         raise HTTPException(status_code=500, detail=str(error))

#     except Exception as error:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(error))

@app.get("/predict-unseen")
def predict_unseen(
    formula: str = Query(..., min_length=1),
    material_id: str = Query(..., min_length=1)
):
    try:
        print("\n===== UNSEEN PREDICTION STARTED =====")
        print("Formula:", formula)
        print("Material ID:", material_id)

        result = predict_unseen_material(
            electrode_formula=formula,
            selected_material_id=material_id
        )

        print("PREDICTION SUCCESSFUL")
        return result

    except Exception as error:
        print("\n===== UNSEEN PREDICTION ERROR =====")
        traceback.print_exc()
        print("ERROR TYPE:", type(error).__name__)
        print("ERROR MESSAGE:", repr(error))
        print("===================================")

        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {str(error)}"
        )

# =====================================================
# UNSEEN MATERIAL - SHAP ANALYSIS
# =====================================================

@app.get("/explain-unseen")
def explain_unseen_endpoint(
    formula: str = Query(..., min_length=1),
    material_id: str = Query(..., min_length=1)
):
    try:
        local_result = explain_unseen_material(
            electrode_formula=formula,
            selected_material_id=material_id
        )

        global_result = get_unseen_global_shap(
            sample_size=200
        )

        return {
            "status": "success",
            "local": local_result,
            "global": global_result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except LookupError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )



@app.get("/model/evaluation")
def model_evaluation_endpoint():
    return {
        "status": "success",
        "evaluation_method": "GroupShuffleSplit",
        "test_size": 0.20,
        "random_state": 42,
        "group_column": "Material_ID",
        "target": "Voltage_V",
        "unit": "V",
        "models": MODEL_EVALUATION_METRICS
    }
# Serve frontend images, CSS and other static files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Metrics copied from the evaluated notebook's held-out test results.
# Evaluation method: GroupShuffleSplit(test_size=0.20, random_state=42),
# grouping by Material_ID. Target: Voltage_V.
CRYSTALVOLT_TEST_METRICS = {
    "evaluation_split": {
        "method": "GroupShuffleSplit",
        "test_size": 0.20,
        "random_state": 42,
        "group_column": "Material_ID",
        "target": "Voltage_V",
        "unit": "V"
    },
    "models": {
        "Random Forest": {"MAE": 0.6095685513131379, "RMSE": 1.0323270605215529, "R2": 0.6741137972875346},
        "XGBoost": {"MAE": 0.6212714199720999, "RMSE": 1.0225237755714376, "R2": 0.6802738336840959},
        "KRR": {"MAE": 0.71742742147525, "RMSE": 1.075904648621766, "R2": 0.6460198423766286},
        "SVR": {"MAE": 0.657909987171247, "RMSE": 1.1239897083575037, "R2": 0.6136721476442457},
        "DNN": {"MAE": 0.819229323524069, "RMSE": 1.1943656663735032, "R2": 0.5637796002111104}
    },
    "source": "ML_with_SHAP11(1).ipynb, held-out test evaluation cells"
}


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "CrystalVolts API is running"}


# --------------------------------------------------
# DATASET STATUS
# --------------------------------------------------

@app.get("/dataset/status")
def dataset_status_endpoint():
    try:
        return get_dataset_status()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
# --------------------------------------------------
# GET MATERIAL PROPERTIES FROM LOCAL DATASET ONLY
# --------------------------------------------------

@app.get("/dataset/materials/{material_id}")
def get_dataset_material_properties(material_id: str):
    try:
        records = get_dataset_records_by_material_id(material_id)

        if not records:
            raise HTTPException(
                status_code=404,
                detail="Material not found in the local dataset."
            )

        return {
            "status": "success",
            "source": "battery_voltage_dataset_3977.csv",
            "material_id": material_id,
            "record_count": len(records),
            "records": records
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# --------------------------------------------------
# SEARCH MATERIALS
# --------------------------------------------------

@app.get("/materials/search")
def search_materials_endpoint(
    formula: str = Query(..., min_length=1),
    working_ion: str = Query("Li", min_length=1)
):
    try:
        dataset_result = search_dataset_by_formula(formula, working_ion)

        # Formula exists in local ML dataset
        if dataset_result["formula_exists_in_dataset"]:
            if dataset_result["materials"]:
                return {
                    "formula": formula,
                    "working_ion": working_ion,
                    "dataset_found": True,
                    "dataset_has_selected_ion_records": True,
                    "dataset_record_count": dataset_result["matching_record_count"],
                    "total_results": len(dataset_result["materials"]),
                    "materials": dataset_result["materials"],
                    "dataset_records": dataset_result["dataset_records"],
                    "message": "Formula found in dataset. Dataset properties are included."
                }

            return {
                "formula": formula,
                "working_ion": working_ion,
                "dataset_found": True,
                "dataset_has_selected_ion_records": False,
                "dataset_record_count": 0,
                "total_results": 0,
                "materials": [],
                "dataset_records": [],
                "message": (
                    f"Formula exists in dataset, but no records were found "
                    f"for working ion {working_ion}."
                )
            }

        # Formula absent from dataset: search Materials Project
        candidate_materials = search_materials(formula)

        if not candidate_materials:
            return {
                "formula": formula,
                "working_ion": working_ion,
                "dataset_found": False,
                "prediction_available": False,
                "total_results": 0,
                "materials": [],
                "message": (
                    "Formula was not found in the ML dataset or Materials Project."
                )
            }

        # Rank using available stability information
        for material in candidate_materials:
            material["working_ion"] = working_ion
            material["dataset_properties"] = []
            material["dataset_record_count"] = 0
            material["is_recommended"] = False

        stable_candidates = [
            material for material in candidate_materials
            if material.get("is_stable") is True
        ]

        ranking_pool = stable_candidates if stable_candidates else candidate_materials

        ranked_candidates = sorted(
            ranking_pool,
            key=lambda material: (
                material.get("energy_above_hull_eV_per_atom") is None,
                material.get("energy_above_hull_eV_per_atom")
                if isinstance(
                    material.get("energy_above_hull_eV_per_atom"),
                    (int, float)
                )
                else float("inf"),
                str(material.get("material_id", ""))
            )
        )

        recommended = ranked_candidates[0]
        recommended["is_recommended"] = True

        hull = recommended.get("energy_above_hull_eV_per_atom")

        if recommended.get("is_stable") is True:
            reason = (
                "Stable entry in Materials Project; preferred among "
                "the returned records."
            )
        elif isinstance(hull, (int, float)):
            reason = (
                f"Lowest available energy above hull among returned "
                f"candidates ({hull:.4f} eV/atom)."
            )
        else:
            reason = (
                "Selected by deterministic fallback because stability "
                "data was unavailable."
            )

        recommended["recommendation_reason"] = reason

        for material in candidate_materials:
            material.setdefault("recommendation_reason", None)

        return {
            "formula": formula,
            "working_ion": working_ion,
            "dataset_found": False,
            "prediction_available": False,
            "source": "Materials Project",
            "total_results": len(candidate_materials),
            "materials": candidate_materials,
            "recommended_material_id": recommended["material_id"],
            "recommendation_basis": (
                "Materials Project stability fields "
                "(is_stable and energy_above_hull); "
                "not an experimental performance ranking."
            ),
            "message": (
                "Material found in Materials Project. Scientific properties "
                "can be explored, but voltage prediction is unavailable "
                "because the material is not in the current ML training dataset."
            )
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))



@app.get("/explain/plots")
def shap_plot_data_endpoint(
    sample_size: int = Query(200, ge=1, le=3977)
):
    try:
        return get_shap_plot_data(sample_size=sample_size)

    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# --------------------------------------------------
# GET COMPLETE MATERIAL PROFILE
# --------------------------------------------------

@app.get("/materials/{material_id}")
def get_material_endpoint(material_id: str):
    try:
        material = get_material_by_id(material_id)

        if material is None:
            raise HTTPException(status_code=404, detail="Material not found")

        dataset_records = get_dataset_records_by_material_id(material_id)

        material["dataset_information"] = {
            "found_in_dataset": bool(dataset_records),
            "record_count": len(dataset_records),
            "records": dataset_records,
            "source": "battery_voltage_dataset_3977.csv"
        }

        return material

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# --------------------------------------------------
# GET BATTERY INFORMATION
# --------------------------------------------------

@app.get("/materials/{material_id}/battery")
def get_battery_endpoint(
    material_id: str,
    formula: str = Query(..., min_length=1),
    working_ion: str = Query("Li", min_length=1)
):
    try:
        return get_battery_information(
            material_id=material_id,
            formula=formula,
            working_ion=working_ion
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# ==========================================
# FAST MATERIAL PROPERTIES ENDPOINT
# ==========================================

@app.get("/materials/{material_id}/properties")
def get_material_properties_endpoint(material_id: str):

    try:
        profile = get_material_by_id(material_id)

        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Material not found in Materials Project."
            )

        return {
            "status": "success",
            "source": "Materials Project",
            "material_id": material_id,
            "sections": {
                "material_overview": {
                    "title": "Material Overview",
                    **profile.get("identity", {})
                },
                "chemical_composition": {
                    "title": "Chemical Composition",
                    **profile.get("identity", {}).get("composition", {})
                },
                "physical_properties": {
                    "title": "Physical Properties",
                    **profile.get("physical_properties", {})
                },
                "crystal_structure": {
                    "title": "Crystal Structure",
                    **profile.get("crystal_structure", {})
                },
                "atomic_structure": {
                    "title": "Atomic Structure",
                    **profile.get("atomic_structure", {})
                },
                "electronic_properties": {
                    "title": "Electronic Properties",
                    **profile.get("electronic_properties", {})
                },
                "stability": {
                    "title": "Stability",
                    **profile.get("stability_properties", {})
                }
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Materials Project request failed: {error}"
        )
# --------------------------------------------------
# GET DETAILED MATERIAL DESCRIPTION
# --------------------------------------------------

@app.get("/materials/{material_id}/description")
def get_material_description_endpoint(
    material_id: str,
    formula: str = Query(..., min_length=1),
    working_ion: str = Query("Li", min_length=1)
):
    try:
        profile = get_material_by_id(material_id)

        if not profile:
            raise HTTPException(status_code=404, detail="Material not found")

        battery_data = get_battery_information(
            material_id,
            formula,
            working_ion
        )

        literature = build_literature_description(formula, rows=10)

        papers = literature.get("references", [])
        literature["papers"] = get_papers_with_abstracts(papers)

        report = generate_material_description(
            material_profile=profile,
            battery_data=battery_data,
            literature=literature
        )

        return report

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))



# --------------------------------------------------
# PREDICT VOLTAGE USING ELECTRODE FORMULA
# --------------------------------------------------

# --------------------------------------------------
# PREDICT VOLTAGE USING ELECTRODE FORMULA
# --------------------------------------------------

@app.get("/predict")
def predict_voltage_by_formula_endpoint(
    formula: str = Query(..., min_length=1),
    working_ion: str = Query("Li", min_length=1)
):
    try:
        dataset_inputs = get_prediction_inputs_by_formula(
            formula=formula,
            working_ion=working_ion
        )

        # Formula or working ion not found in the ML dataset.
        # Return an empty result so the frontend can start
        # the unseen-material workflow.
        if not dataset_inputs:
            return {
                "status": "success",
                "formula": formula,
                "working_ion": working_ion,
                "input_source": "battery_voltage_dataset_3977.csv",
                "total_matching_records": 0,
                "results": [],
                "message": (
                    "Material not found in the training dataset. "
                    "Continue with Materials Project host structures."
                )
            }

        all_predictions = []

        for dataset_input in dataset_inputs:
            prediction_result = predict_voltage(
                dataset_input["input_features"]
            )

            all_predictions.append({
                "material_id": dataset_input["material_id"],
                "formula": dataset_input["formula"],
                "working_ion": dataset_input["working_ion"],
                "input_feature_count": dataset_input["input_feature_count"],
                "processed_feature_count": prediction_result["processed_feature_count"],
                "predictions": prediction_result["predictions"],
                "unit": "V"
            })

        return {
            "status": "success",
            "formula": formula,
            "working_ion": working_ion,
            "input_source": "battery_voltage_dataset_3977.csv",
            "total_matching_records": len(all_predictions),
            "results": all_predictions
        }

    except HTTPException:
        raise

    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))
# --------------------------------------------------
# GENERAL PREDICTION USING 44 INPUT FEATURES
# --------------------------------------------------

@app.post("/predict")
def predict_voltage_endpoint(input_features: dict = Body(...)):
    try:
        return predict_voltage(input_features)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# --------------------------------------------------
# GLOBAL SHAP EXPLANATION
# --------------------------------------------------

@app.get("/explain/global")
def global_shap_endpoint(
    sample_size: int = Query(200, ge=1, le=3977)
):
    try:
        return explain_global(sample_size=sample_size)

    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# --------------------------------------------------
# LOCAL SHAP EXPLANATION
# --------------------------------------------------

@app.get("/explain/{material_id}")
def local_shap_endpoint(
    material_id: str,
    working_ion: str = Query("Li")
):
    try:
        return explain_material(
            material_id=material_id,
            working_ion=working_ion
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))