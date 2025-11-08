# ============================================================
# MVR Estimation Backend 
# ============================================================

# -----------------------------
# Required Imports
# -----------------------------
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import joblib
from pathlib import Path
import json
import sklearn
import sys
print(sys.executable)


# ============================================================
# Configure file paths for pre-trained model (pickle files)
# ============================================================

from pathlib import Path

MODEL_DIR = Path("models")
for f in ["MVR_DFC_model_final_nov_2025.pkl",
          "MVR_GM_model_final_nov_2025.pkl",
          "MVR_FCP_model_final_nov_2025.pkl"]:
    path = MODEL_DIR / f
    print(f"{f} exists: {path.exists()}")

# Model filenames
DFC_MODEL_NAME = "MVR_DFC_model_final_nov_2025.pkl"
GM_MODEL_NAME  = "MVR_GM_model_final_nov_2025.pkl"
FCP_MODEL_NAME = "MVR_FCP_model_final_nov_2025.pkl"

# ------------------------------------------------------------
# Load model lists from pickle/joblib files
# ------------------------------------------------------------
def load_model_list(filename):
    """
    Load a pickle or joblib model list/dict from file.
    Returns a list of model objects that support .predict().
    """
    file_path = MODEL_DIR / filename

    # Validate file extension and existence
    if not filename.endswith(".pkl"):
        raise ValueError(f"{filename} is not a .pkl file. Only pickle files are supported.")
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    # Try pickle first, then joblib
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as p_err:
        try:
            obj = joblib.load(file_path)
        except Exception as j_err:
            raise RuntimeError(f"Failed to load {file_path}: pickle error: {p_err}; joblib error: {j_err}")

    # Normalize to list of models
    if isinstance(obj, dict):
        model_list = list(obj.values())
    elif isinstance(obj, list):
        model_list = obj
    else:
        raise ValueError(f"{filename} must contain a list or dict of model objects.")

    return [m for m in model_list if hasattr(m, "predict")]


# ------------------------------------------------------------
# Safely load all model groups with logging
# ------------------------------------------------------------
def safe_load_models():
    """Attempt to load all models, log failures instead of failing silently."""
    loaded = {}
    for name in [DFC_MODEL_NAME, FCP_MODEL_NAME, GM_MODEL_NAME]:
        try:
            models = load_model_list(name)
            loaded[name] = models
            print(f"Loaded {len(models)} model(s) from {name}")
        except Exception as e:
            print(f"Error loading {name}: {type(e).__name__}: {e}")
            loaded[name] = []
    return loaded


# Load models
_loaded     = safe_load_models()
DFC_MODELS  = _loaded.get(DFC_MODEL_NAME, [])
FCP_MODELS  = _loaded.get(FCP_MODEL_NAME, [])
GM_MODELS   = _loaded.get(GM_MODEL_NAME, [])

# Validate successful model load
if not DFC_MODELS or not GM_MODELS or not FCP_MODELS:
    raise RuntimeError("One or more model lists are empty. Check that the underlying model .pkl files loaded correctly.")


# ============================================================
# Model Features & Configuration
# ============================================================

# --- Feature lists for each submodel ---
FCP_FEATURES = [
    'mvr_startup_score_binary', 'Reseller/VAR', 'Data_Services', 'Hardware', 'Software',
    'Labor_Services', 'Engineering_and_Advisory_Services', 'company_labor_class',
    'company_size_medium', 'has_sat'
]

DFC_FEATURES = [
    'Data_Services', 'Hardware', 'Asset-Intensive', 'has_sat',
    'company_labor_class', 'company_size_medium'
]

GM_FEATURES = [
    'Reseller/VAR', 'Data_Services', 'Infrastructure_Services', 'Hardware', 'Software',
    'Asset-Intensive', 'Labor_Services', 'Engineering_and_Advisory_Services',
    'mvr_startup_score_binary', 'has_sat', 'company_labor_class'
]

# --- Column renaming map ---
COLUMN_RENAME_MAP = {                                         ############## ROS PLEASE ENSURE THAT THE FRONTEND PASSES VALID (MATCHING) KEYS, IF THERE'S A MISMATCH THEN OUTPUT COULD BE AN UNEXPECTED VALUE
    "Company Name": "company_name",
    "name": "company_name",
    "Year Founded": "year_founded",
    "founded_year": "year_founded",
    "Headcount": "headcount",
    "Trading Status": "trading_status",
    "Reseller/VAR": "Reseller/Var",
    "Data Services": "Data_Services",
    "Infrastructure Services": "Infrastructure_Services",
    "Hardware": "Hardware",
    "Software": "Software",
    "Asset-Intensive": "Asset-Intensive",
    "Labor Services": "Labor_Services",
    "Engineering & Advisory Services": "Engineering_and_Advisory_Services",
    "Company Labor Class": "company_labor_class",
    "Satellite Owner": "has_sat",
}

# --- Global constants ---
FCP_MODEL_RETURN_UNITS = 1_000   # FC/P model returns values in thousands of USD
STDEV_PCT_ERROR = 0.48409        # 1 std deviation of percent error (empirical calibration)


# ============================================================
# Section 3: Utility Functions
# ============================================================
def fmt_usd(x):
    """Format numeric value as USD ($1,234)."""
    return f"${x:,.0f}" if np.isfinite(x) else "NaN"


def fmt_pct(x):
    """Format numeric value as percentage (e.g., 45%)."""
    return f"{x:.0f}%" if np.isfinite(x) else "NaN"


def standardize_input_keys(inputs):
    """Standardize column names for dict or DataFrame inputs."""
    if isinstance(inputs, pd.DataFrame):
        df = inputs.copy()
        df.columns = [COLUMN_RENAME_MAP.get(c.strip(), c.strip()) for c in df.columns]
        return df
    elif isinstance(inputs, dict):
        return {COLUMN_RENAME_MAP.get(k.strip(), k.strip()): v for k, v in inputs.items()}
    else:
        raise TypeError("Input must be a dict or pandas DataFrame")


def truemvr(fc, d, gm):
    """Compute true MVR. Returns NaN if fc or gm is zero."""
    fc, d, gm = float(fc), float(d), float(gm)
    if fc == 0 or gm == 0:
        return np.nan
    return (fc * (1 + (d / fc))) / gm


def align_to_schema(df, feature_list):
    """Ensure DataFrame columns match model feature schema."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df_aligned = df.reindex(columns=feature_list, fill_value=0)
    for col in feature_list:
        df_aligned[col] = pd.to_numeric(df_aligned[col], errors="coerce").fillna(0)
    return df_aligned


def force_schema(fcp_passed, dfc_passed, gm_passed):
    """Force all inputs to conform to submodel feature schemas."""
    return (
        align_to_schema(fcp_passed, FCP_FEATURES),
        align_to_schema(dfc_passed, DFC_FEATURES),
        align_to_schema(gm_passed, GM_FEATURES)
    )


# ============================================================
# Startup Score Calculation Function.  
# If current_year is not passed, then it uses the current year
# ============================================================
def calculate_mvr_startup_score_binary(year_founded, headcount, company_status, current_year=None):
    """Compute binary startup score based on company age and headcount."""

    try:
        current_year = datetime.now().year if current_year is None else current_year

        if year_founded is None or headcount is None or company_status is None:
            print("Warning - startup score error, returning neutral value of 0.5")
            return 0.5

        if headcount <= 0 or year_founded <= 0 or year_founded > current_year:
            print("Warning - startup score error, returning neutral value of 0.5")
            return 0.5

        years_since_founded = current_year - year_founded
        age_score = np.exp(-(np.log(5) / 25) * years_since_founded)
        headcount_size_score = 0.01 + (0.99 / (1 + (headcount / 200) ** 1.03))

        h, a = headcount_size_score, age_score
        if abs(h - a) <= 0.5:
            parent_score = np.sqrt(h * a) * 0.5 + a * 0.5
        else:
            parent_score = np.sqrt(h * a)

        if str(company_status).lower() == "public":
            parent_score *= 0.95

        if not np.isfinite(parent_score):
            return 0
        return float(np.clip(parent_score, 0, 1))

    except Exception as e:
        print(f"Exception {e}!")
        return 0


# ============================================================
# Core MVR Calculator
# ============================================================
def mvr_calculator(inputs_dict, DFCmodels, GMmodels, FCPmodels):
    """
    Computes MVR for a single company using dictionary input. This is the
    workhorse function for estimating MVR and the 3 sub-models.
    
    Parameters:   inputs_dict:  A dictionary containing the input features for a single target company
                  DFCmodels, GMmodels, FCPmodels:  Reference to the pre-loaded models to apply for the prediction
    
    Return:       dict containing the following:
                       mvr_pred:   MVR prediction in whole dollars (for example, $1,234,456 is the return value for MVR of $1.2 million
                       d/fc_pred:  D/FC prediction (ratio)
                       gm_pred:    GM prediction (0-1)
                       fc/p_pred:  fixed costs per head (FC/P) prediction in whole dollars (for example, $123,456 is the return for $123K)
                       mvr_startup_score:  Continuous startup score (0-1)
                       mvr_startup_score_binary:  Binary startup score (0 or 1)
                       headcount: Headcount value with error-checking

    """

    # --- Standardize keys ---
    inputs = standardize_input_keys(inputs_dict.copy())

    # --- Safe numeric casting ---
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    # --- Extract basic parameters ---
    year_founded = safe_float(inputs.get("year_founded"))
    headcount    = safe_float(inputs.get("headcount"))
    trading_status = inputs.get("trading_status", "Private")

    if headcount <= 0:
        print("Warning: Headcount is zero or negative, setting to 1 to avoid errors.")
        headcount = 1

    # --- Compute startup score ---
    mvr_startup_score = calculate_mvr_startup_score_binary(year_founded, headcount, trading_status)
    mvr_startup_score_binary = int(mvr_startup_score >= 0.5)

    # --- Handle labor class mapping (A–E → 0–4) ---
    if "company_labor_class" in inputs:
        val = inputs["company_labor_class"]
        if isinstance(val, str):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            inputs["company_labor_class"] = mapping.get(val.upper(), 0)
        inputs["company_labor_class"] = safe_float(inputs["company_labor_class"])

    # --- Feature setup ---
    features = inputs.copy()
    features["mvr_startup_score_binary"] = mvr_startup_score_binary
    features["headcount_log"] = np.log(max(headcount, 1))
    features["company_size_medium"] = int(74 < headcount <= 749)
    features["company_size_large"]  = int(headcount > 749)

    # --- Build DataFrames for submodels ---
    dfc_X = pd.DataFrame([[features.get(f, 0) for f in DFC_FEATURES]], columns=DFC_FEATURES)
    gm_X  = pd.DataFrame([[features.get(f, 0) for f in GM_FEATURES]], columns=GM_FEATURES)
    fcp_X = pd.DataFrame([[features.get(f, 0) for f in FCP_FEATURES]], columns=FCP_FEATURES)

    # --- Submodel predictions ---
    DFC_preds = np.column_stack([m.predict(dfc_X) for m in DFCmodels])
    GM_preds  = np.column_stack([m.predict(gm_X)  for m in GMmodels])
    FCP_preds = np.column_stack([np.expm1(m.predict(fcp_X)) for m in FCPmodels])

    # --- Median ensemble (apply floors) ---
    dfc_pred = float(np.median(np.maximum(DFC_preds, 0.00)))
    gm_pred  = float(np.median(np.maximum(GM_preds,  0.05)))
    fcp_pred = float(np.median(np.maximum(FCP_preds, 8)))

    # --- Scale FCP output (thousands → full dollars) ---
    fcp_pred *= FCP_MODEL_RETURN_UNITS

    # --- Compute final MVR prediction ---
    mvr_pred = (fcp_pred * headcount * (1 + dfc_pred)) / gm_pred

    # --- Return output dictionary ---
    return {
        "mvr_pred": mvr_pred,
        "d/fc_pred": dfc_pred,
        "gm_pred": gm_pred,
        "fc/p_pred": fcp_pred,
        "mvr_startup_score": mvr_startup_score,
        "mvr_startup_score_binary": mvr_startup_score_binary,
        "headcount": headcount,
    }


# ============================================================
# Compute MVR Wrappers
# ============================================================


def compute_mvr(inputs_dict, current_year=None, add_debug_noise=False, scale_display_units=1000.0):
    """
    Wrapper around mvr_calculator that handles user inputs, 
    scaling, overrides, and formatted output. It assumes that the pickle models have already been loaded in the 
    environment before this function can be succesfully called.
    
    Parameters:   inputs_dict:  A dictionary containing the input features for a single target company
                  current_year: The current year (if predicting the current MVR). By default it will model based on current year.
                  add_debug_noise: Deprecated
                  scale_display_units:  For the 'display' dict nested within the return, this indicates the units to display. For example, 1000 
                                        will output values in $000s, 1000_000 will output values in $millions, etc.
    
    Return:       dict containing the following:

                        "headcount": headcount of the target company with error-checking,
                        "dfc_pred": the estimated D/FC ratio,
                        "fcp_pred": the estimated FC/P (i.e. fixed costs per person), in whole dollars (e.g., $123,000 is the return for $123K)
                        "gm_pred": the estimated gross margin (0.0-1.0)
                        "fcp_total_usd": Same as fcp_pred
                        "dfc_user": user's input for d/fc if available otherwise None. Note that if user provided a value, it overrides the model prediction for that metric
                        "fcp_user": user's input for fc/p if available otherwise None. Note that if user provided a value, it overrides the model prediction for that metric
                        "gm_user": user's input for gm if available, otherwise None. Note that if user provided a value, it overrides the model prediction for that metric
                        "normalized_capex_total_usd": The total normalized capex in whole dollars
                        "fixed_costs_total_usd": The total fixed costs in whole dollars (i.e., equal to fc/p * headcount)
                        "gm_percent": Equals gm expressed as a value, i.e., 0.50 would be represented in this return as 50, representing 50 percent
                        "mvr_value": Predicted MVR in whole dollars
                        "error": MVR error (1 stddev) in whole dollars
                        "overall_error_usd": Same as above
                        "mvr_value_display": A formatted version of MVR but still in whole dollars (e.g., $12,345,678 would return for ~$12 million) 
                        "overall_error_display": A formatted version of the error but still in whole dollars
                        "display": {
                            "Normalized Capex (DFC)":DEPRECATED, SEE COMMENT BELOW AND ADJUST IN FRONT END
                            "Normalized Capex (Total)": Normalized capex displayed as a string with a $' and in whatever units were passed to the function as scale_display_units
                            "D/FC Ratio": String version of D/FC shown to 2-decimals
                            "Steady-State Fixed Costs (Total)": Steady-state Fixed Costs total, displayed as $ with whatever units were passed
                            "Steady-State Fixed Costs (Per Person): Steady-state Fixed Costs per person, displayed as $ with whatever units were passed
                            "MVR": Best MVR estimate, displayed as $ with whatever untis were passed
                            "Gross Margin (GM)": Gross margin displayed as a percentage
                        },

    """

    if isinstance(inputs_dict, pd.Series):
        inputs_dict = inputs_dict.to_dict()
    if not isinstance(inputs_dict, dict):
        raise TypeError(f"Expected dict, got {type(inputs_dict)}")

    results = mvr_calculator(inputs_dict, DFC_MODELS, GM_MODELS, FCP_MODELS)
    headcount = results["headcount"]

    # --- Extract user-specified values (if provided) ---
    user_normalized_capex = inputs_dict.get("Normalized Capex", None)
    user_fcp = inputs_dict.get("Steady-State Fixed Costs", None)
    user_gm  = inputs_dict.get("Gross Margin %", None)
    user_dfc = user_normalized_capex / (user_fcp * headcount) if user_normalized_capex is not None and user_fcp is not None else None

    # --- Encode labor class if applicable ---
    if "company_labor_class" in inputs_dict:
        val = inputs_dict["company_labor_class"]
        if isinstance(val, str):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            inputs_dict["company_labor_class"] = mapping.get(val.strip().upper(), 0)
        else:
            try:
                inputs_dict["company_labor_class"] = float(val)
            except (TypeError, ValueError):
                inputs_dict["company_labor_class"] = 0

    # --- Use user values if available ---
    fcp_pred = results["fc/p_pred"] if user_fcp is None else float(user_fcp)
    total_fixed_costs = fcp_pred * headcount
    gm_pred = results["gm_pred"] if user_gm is None else float(user_gm)
    dfc_pred = results["d/fc_pred"] if user_normalized_capex is None else float(user_normalized_capex / total_fixed_costs)

    # --- Derived quantities ---
    normalized_capex = total_fixed_costs * dfc_pred
    gm_percent = gm_pred * 100.0
    mvr_pred = (total_fixed_costs + normalized_capex) / gm_pred
    overall_error = mvr_pred * STDEV_PCT_ERROR

    # --- Build result dictionary ---
    return {
        "headcount": headcount,
        "dfc_pred": dfc_pred,
        "fcp_pred": fcp_pred,
        "gm_pred": gm_pred,
        "dfc_total_usd": -9999.999,    # DEPRECATED - REMOVE IN FRONTEND.  D/FC IS NOT A VALUE THAT IS IN DOLLARS, IT IS A RATIO THAT DIVIDES NORMALIZED CAPEX ("D") BY STEADY-STATE FIXED COSTS ("FC")
        "fcp_total_usd": fcp_pred,
        "gm_total_usd": -9999.999,   # DEPRECATED - REMOVE IN FRONTEND. GM IS A RATIO NOT A DOLLAR VALUE
        "dfc_sub": [],
        "fcp_sub": [],
        "gm_sub": [],
        "dfc_user": user_dfc,
        "fcp_user": user_fcp,
        "gm_user": user_gm,
        "normalized_capex_total_usd": normalized_capex,
        "fixed_costs_total_usd": total_fixed_costs,
        "gm_percent": gm_percent,
        "mvr_value": mvr_pred,
        "error": overall_error,
        "overall_error_usd": overall_error,
        "mvr_value_display": f"${mvr_pred:,.2f}",
        "overall_error_display": f"${overall_error:,.2f}",
        "display": {
            # "Normalized Capex (DFC)": fmt_usd(-9999.999),     # NORMALIZED CAPEX TOTAL IS NOW RETURNED BELOW, AND D/FC RATIO IS ALSO NOW RETURNED BELOW, BUT NORMALIZED CAPEX IS A DOLLAR AMOUNT WHEREAS D/FC IS A RATIO, SO THIS ITEM IS A MISMASH OF TWO DIFFERENT THINGS, SO RETURNS A PLACEHOLDER VALUE AND WILL NEED TO BE REMOVED
            "Normalized Capex (Total)": fmt_usd(normalized_capex / scale_display_units),
            "D/FC Ratio": f"{dfc_pred:.2f}",
            "Steady-State Fixed Costs (Total)": fmt_usd(total_fixed_costs / scale_display_units),
            "Steady-State Fixed Costs (Per Person)": fmt_usd(fcp_pred / scale_display_units),
            "MVR": fmt_usd(mvr_pred / scale_display_units),
            "Gross Margin (GM)": fmt_pct(gm_percent)
        },
    }


# ============================================================
# Function allows for batch processing of one more more target companies/records
# ============================================================
def compute_mvr_batch(inputs_list, current_year=None, add_debug_noise=False, scale_display_units=1000.0):
    """
    Compute MVR for one or more companies.
    Handles single dict, list of dicts, or DataFrame input.
    Same input handling as compute_mvr.
    Returns a list of output dicts containing the same keys as compute_mvr (see compute_mvr)
    """
    if isinstance(inputs_list, dict):
        inputs_list = [inputs_list]
    elif isinstance(inputs_list, pd.DataFrame):
        inputs_list = inputs_list.to_dict(orient="records")

    results = []
    for i, company_inputs in enumerate(inputs_list):
        if isinstance(company_inputs, pd.Series):
            company_inputs = company_inputs.to_dict()
        elif not isinstance(company_inputs, dict):
            raise TypeError(f"Each input must be a dict or pandas Series, got {type(company_inputs)}")

        res = compute_mvr(company_inputs, current_year=current_year, add_debug_noise=add_debug_noise, scale_display_units=scale_display_units)
        res["company_name"] = (
            company_inputs.get("Company Name")
            or company_inputs.get("company_name")
            or company_inputs.get("name")
            or f"Company {i+1}"
        )
        results.append(res)

    return results


# ============================================================
# Example usage for testing/debugging purposes
# ============================================================

print(f"scikit-learn version: {sklearn.__version__}")

inputs_list = [
    {
        "Company Name": "ExoAnalytic Solutions",
        "Year Founded": 2008,
        "headcount": 115,
        "Trading Status": "Private",
        "Reseller/VAR": 0,
        "Data Services": 1,
        "Infrastructure Services": 0,
        "Hardware": 0,
        "Software": 1,
        "Asset-Intensive": 0,
        "Labor Services": 0,
        "Engineering & Advisory Services": 1,
        "Company Labor Class": 1,
        "Satellite Owner": 0,
    },
    {
        "Company Name": "Turion",
        "Year Founded": 2020,
        "headcount": 140,
        "Trading Status": "Private",
        "Reseller/VAR": 0,
        "Data Services": 0,
        "Infrastructure Services": 0,
        "Hardware": 1,
        "Software": 1,
        "Asset-Intensive": 1,
        "Labor Services": 0,
        "Engineering & Advisory Services": 1,
        "Company Labor Class": 1,
        "Satellite Owner": 1,
    }
]

current_year = 2025

test_from_full_ssa_inputs = False  # False just to test the examples that are hardcoded above, True if you want to test an Excel file that matches the SSA input file format

if test_from_full_ssa_inputs == True:
    ssadf = pd.read_excel(r"C:\Users\jdcad\Downloads\SSA Companies MVR Predictors (2025-11-06 Final).xlsx")
    inputs_list = ssadf.to_dict(orient="records")
    results = compute_mvr_batch(inputs_list, current_year=current_year, add_debug_noise=False)
else:
    results = compute_mvr_batch(inputs_list, current_year=current_year, add_debug_noise=False)

print("\n ----- OUTPUT FROM BACKEND ------ \n")
print(json.dumps(results, indent=4))