## Annotated model_backend.py for OCEA team
## Date: 30 October 2025

# Load required libraries
# import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------
# Section 1: Load Pre-Trained Models
# -----------------------------

# Name the folder where the models are stored
MODEL_DIR = Path("models")

# Retrieve the models
def load_model_list(filename):
    if not filename.endswith(".pkl"):
        raise ValueError(f"{filename} is not a .pkl file. Only pickle files are supported.")

    with open(MODEL_DIR / filename, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        model_list = list(obj.values())
    elif isinstance(obj, list):
        model_list = obj
    else:
        raise ValueError(f"{filename} must contain a list or dict of model objects.")

    return [m for m in model_list if hasattr(m, "predict")]

try:
    DFC_MODELS = load_model_list("DFCmodels5strong.pkl")
    FCP_MODELS = load_model_list("FCPmodels5strong.pkl")
    GM_MODELS  = load_model_list("GMmodels5strong.pkl")
except Exception:
    DFC_MODELS = []
    FCP_MODELS = []
    GM_MODELS  = []

########### OLD CODE ################

# Name the folder where the models are stored
# MODEL_DIR = Path("models")

# Retrieve the models
# def load_model_list(filename):
#     if not filename.endswith(".pkl"):
#         raise ValueError(f"{filename} is not a .pkl file. Only pickle files are supported.")
#     obj = joblib.load(MODEL_DIR / filename)
#     if isinstance(obj, dict):
#         model_list = list(obj.values())
#     elif isinstance(obj, list):
#         model_list = obj
#     else:
#         raise ValueError(f"{filename} must contain a list or dict of model objects.")
#     return [m for m in model_list if hasattr(m, "predict")]

# try:
#     DFC_MODELS = load_model_list("DFCmodels5strong.pkl")
#     FCP_MODELS = load_model_list("FCPmodels5strong.pkl")
#     GM_MODELS  = load_model_list("GMmodels5strong.pkl")
# except Exception:
#     DFC_MODELS = []
#     FCP_MODELS = []
#     GM_MODELS  = []

# -----------------------------
# Section 2: Define Model Features
# -----------------------------

# D/FC model for Normalized Capex
DFC_FEATURES = [
    'Data_Services','Hardware','Asset-Intensive','has_sat',
    'company_labor_class','company_size_medium'
]

# Fixed Costs per Person Model
FCP_FEATURES = [
    'is_startup','Reseller/VAR','Data_Services','Hardware','Software',
    'Labor_Services','Engineering_and_Advisory_Services',
    'company_labor_class','company_size_medium','has_sat'
]

# Gross Margin % Model
GM_FEATURES = [
    'Reseller/VAR','Data_Services','Infrastructure_Services','Hardware','Software',
    'Asset-Intensive','Labor_Services','Engineering_and_Advisory_Services',
    'mvr_startup_score_binary','has_sat','company_labor_class'
]

# -----------------------------
# Section 3: Define mvr_startup_score_binary 
# -----------------------------

# Define the mvr_startup_score_binary from user inputs
def calculate_mvr_startup_score_binary(year_founded, headcount, company_status):
    try:
        current_year = datetime.now().year
        if year_founded is None or headcount is None or company_status is None:
            return 0
        if headcount <= 0 or year_founded <= 0 or year_founded > current_year:
            return 0

        years_since_founded = current_year - year_founded
        age_score = np.exp(-(np.log(5)/25) * years_since_founded)
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
    except Exception:
        return 0

# -----------------------------
# Section 4: Helper Functions
# -----------------------------

# Encode the company_labor_class in backend
def encode_inputs(inputs_dict):
    df = pd.DataFrame([inputs_dict])
    if "company_labor_class" in df.columns:
        df["company_labor_class"] = df["company_labor_class"].map({"A":1,"B":2,"C":3}).fillna(0)
    return df.fillna(0)

# Return the median prediction from dependency models
def predict_group(models, feature_names, inputs_dict, add_noise=False):
    df = encode_inputs(inputs_dict)
    X_base = df.reindex(columns=feature_names, fill_value=0).to_numpy()

    preds = []
    for i, m in enumerate(models):
        X = X_base
        if add_noise:
            rng = np.random.RandomState(seed=42 + i)
            X = X_base + rng.normal(0, 1e-6, X_base.shape)
        try:
            p = m.predict(X)[0]
        except Exception:
            p = float("nan")
        preds.append(float(p))

    if len(preds) == 0:
        return 0.0, []

    median_pred = float(np.nanmedian(preds))
    return median_pred, preds

# Return the median error from predictions
def calculate_group_error(preds, user_value=None, scale=1.0):
    if user_value is not None or len(preds) == 0:
        return 0.0
    median_pred = np.nanmedian(preds)
    errs = [abs(p - median_pred) for p in preds if np.isfinite(p)]
    return float(np.mean(errs) * scale)

# -----------------------------
# Define MVR Computation
# -----------------------------

# Define how MVR is calculated
def compute_mvr(inputs_dict, current_year=None, add_debug_noise=False):
    now_year = current_year if current_year else datetime.now().year
    inputs = dict(inputs_dict)

    # Function to avoid error
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    year_founded = safe_float(inputs.get("Year Founded"))
    headcount = safe_float(inputs.get("Headcount"), 1.0)
    trading_status = inputs.get("Trading Status")

    # Compute dynamic mvr_startup_score_binary
    mvr_startup_score_binary = calculate_mvr_startup_score_binary(
        year_founded=year_founded,
        headcount=headcount,
        company_status=trading_status
    )

    # Pass mvr_startup_score_binary to GM model
    gm_inputs = inputs.copy()
    gm_inputs["mvr_startup_score_binary"] = mvr_startup_score_binary

    # Ensemble predictions
    dfc_pred, dfc_sub = predict_group(DFC_MODELS, DFC_FEATURES, inputs, add_noise=add_debug_noise)
    fcp_pred, fcp_sub = predict_group(FCP_MODELS, FCP_FEATURES, inputs, add_noise=add_debug_noise)
    gm_pred, gm_sub   = predict_group(GM_MODELS,  GM_FEATURES,  gm_inputs, add_noise=add_debug_noise)

    # Choose the user value instead of model predictions if the former exists
    user_dfc = inputs_dict.get("Normalized Capex")
    user_fcp = inputs_dict.get("Steady-State Fixed Costs")
    user_gm  = inputs_dict.get("Gross Margin %")

    if user_dfc is not None:
        dfc_pred = float(user_dfc) / 1000.0
    if user_fcp is not None:
        fcp_pred = float(user_fcp) / 1000.0
    if user_gm is not None:
        gm_pred = float(user_gm) / 100.0

    # Scale model outputs to actual financial units
    DFC_SCALE = 10_000.0
    FCP_SCALE = 10_000.0
    GM_SCALE = 100.0

    # Avoid error
    dfc_pred = safe_float(dfc_pred)
    fcp_pred = safe_float(fcp_pred)
    gm_pred = safe_float(gm_pred)

    # Convert to dollar terms for frontend clarity
    dfc_total_usd = dfc_pred * headcount * DFC_SCALE
    fcp_total_usd = fcp_pred * headcount * FCP_SCALE
    gm_percent = gm_pred * GM_SCALE

    # Compute MVR in dollar terms
    try:
        mvr_value = ((fcp_pred * headcount * FCP_SCALE) * (1 + dfc_pred)) / gm_pred
    except Exception:
        mvr_value = float("nan")

    # Assign error estimates per model group
    dfc_error_raw = calculate_group_error(dfc_sub, user_dfc, scale=1000.0)
    fcp_error_raw = calculate_group_error(fcp_sub, user_fcp, scale=1000.0)
    gm_error_raw  = calculate_group_error(gm_sub,  user_gm,  scale=1000.0)

    # Convert errors to same display scale as their respective dependency model
    dfc_error_usd = dfc_error_raw * headcount * DFC_SCALE
    fcp_error_usd = fcp_error_raw * headcount * FCP_SCALE
    gm_error_pct = gm_error_raw
    overall_error_usd = np.nanmean([dfc_error_usd, fcp_error_usd])

    # Format for display
    def fmt_usd(x):
        return f"${x:,.2f}" if np.isfinite(x) else "NaN"
    
    def fmt_pct(x):
        return f"{x:.2f}%" if np.isfinite(x) else "NaN"

    mvr_value_display = fmt_usd(mvr_value)
    overall_error_display = fmt_usd(overall_error_usd)

    # Return results
    return {
        "dfc_pred": dfc_pred,
        "fcp_pred": fcp_pred,
        "gm_pred": gm_pred,

        "dfc_sub": [safe_float(x) for x in dfc_sub],
        "fcp_sub": [safe_float(x) for x in fcp_sub],
        "gm_sub":  [safe_float(x) for x in gm_sub],

        "dfc_user": user_dfc,
        "fcp_user": user_fcp,
        "gm_user": user_gm,

        "dfc_total_usd": dfc_total_usd,
        "fcp_total_usd": fcp_total_usd,
        "gm_percent": gm_percent,

        "headcount": headcount,
        "mvr_value": mvr_value,
        "mvr_value_display": mvr_value_display,
        "error": overall_error_usd,
        "error_display": overall_error_display,

        "submodel_errors": {
            "Normalized Capex (DFC)": fmt_usd(dfc_error_usd),
            "Steady-State (FCP)": fmt_usd(fcp_error_usd),
            "Gross Margin (GM)": fmt_pct(gm_error_pct)
        },

        "display": {
            "Normalized Capex (DFC)": fmt_usd(dfc_total_usd),
            "Steady-State Fixed Costs (Total)": fmt_usd(fcp_total_usd),
            "Steady-State Fixed Costs (Per Person)": fmt_usd(fcp_pred * FCP_SCALE),
            "Gross Margin (GM)": fmt_pct(gm_percent)
        },

        "debug_features": {
            "Years Active": int(safe_float(inputs.get("Years Active", 0))),
            "mvr_startup_score_binary": float(mvr_startup_score_binary)
        }
    }
