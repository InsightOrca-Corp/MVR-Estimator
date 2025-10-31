# -----------------------------
# Import Libraries
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from model_backend_v3 import compute_mvr
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Section 1: Page Configuration
# -----------------------------

# Display the InsightOrca logo
logo=Image.open("InsightOrca-standard.png")
st.image(logo, width=200)

# Single-page layout with headers and description
st.set_page_config(page_title="MVR Estimator", layout="centered")
st.title("Minimum Viable Revenue (MVR) Estimation Tool")
st.markdown(
    "Estimate the **Minimum Viable Revenue (MVR)** to assess company and overall market health using financial and/or non-financial indicators. Please see the Documentation for details."
)

# -----------------------------
# Section 2: Company Information Input
# -----------------------------

# Ask the user if they want to upload a file
use_file = st.radio("Load company data from a file?", ("Yes", "No"), horizontal=True, key="use_file")

# Initiate user form
with st.form("mvr_form"):
    company_data = {}

    # If the user wants to upload a file:
    if use_file == "Yes":
        st.subheader("Upload Your File")
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="uploaded_file")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())
                st.session_state["data_loaded"] = True
            except Exception as e:
                st.error(f"Error reading file: {e}. Please try again")

    # If the user does not want to upload a file:
    else:
        st.subheader("Step 1: Enter Company Information")
        col1, col2 = st.columns(2)

        # First column inputs
        with col1:
            company_name = st.text_input("Company Name (optional)", key="company_name")
            company_data["Year Founded"] = st.number_input(
                "Year Founded", 1800, datetime.now().year, 2015, key="year_founded"
            )
            company_data["Trading Status"] = st.selectbox(
                "Trading Status", ["Public", "Private"], key="trading_status"
            )

        # Second column inputs
        with col2:
            company_data["Headcount"] = st.number_input(
                "Number of Employees", min_value=1, step=1, value=10, key="headcount"
            )
            segments = st.multiselect(
                "Revenue Driver Attributes",
                [
                    "Asset-Intensive", "Engineering & Advisory", "Data Services",
                    "Infrastructure Services", "Labor", "Hardware", "Software", "Reseller/VAR"
                ],
                key="segments"
            )

            # Encode selected segments as binary flags
            for seg in segments:
                key = seg.replace(" & ", "_and_").replace(" ", "_")
                company_data[key] = 1

            # Ensure all expected features exist with default 0
            all_features = [
                "Asset-Intensive", "Engineering_and_Advisory_Services", "Data_Services",
                "Infrastructure_Services", "Labor_Services", "Hardware", "Software", "Reseller/VAR",
                "has_sat", "company_labor_class", "company_size_medium", "is_startup", "mvr_startup_score_binary"
            ]
            for feat in all_features:
                if feat not in company_data:
                    company_data[feat] = 0

            company_data["Years Active"] = datetime.now().year - company_data["Year Founded"]

        # Create financial information section
        st.subheader("Step 2: Enter Financial Information (optional)")
        st.markdown("All units must be in US$ million.")

        has_financials = st.radio(
            "Is Gross Margin, Normalized Capex, and/or Steady-State Fixed Costs known?",
            ("Yes", "No"),
            horizontal=True,
            key="has_financials"
        )

        if has_financials == "Yes":
            col1, col2 = st.columns(2)
            with col1:
                gm_percent = st.number_input("Gross Margin Percentage (%)", 0.0, 100.0, 0.0, 0.1, key="gm_percent")
                norm_capex = st.number_input("Normalized Capex (in US$ millions)", 0.0, step=1000.0, key="norm_capex")
            with col2:
                steady_state_fc = st.number_input("Steady-State Fixed Costs (in US$ millions)", 0.0, step=1000.0, key="steady_fc")
        else:
            gm_percent = norm_capex = steady_state_fc = None

    # Ensure "Estimate MVR" button is at the bottom of the form
    submitted = st.form_submit_button("Estimate MVR")

    if submitted:
        inputs = {**company_data}

        # Optional user overrides
        if gm_percent and gm_percent > 0:
            inputs["Gross Margin %"] = gm_percent
        if norm_capex and norm_capex > 0:
            inputs["Normalized Capex"] = norm_capex
        if steady_state_fc and steady_state_fc > 0:
            inputs["Steady-State Fixed Costs"] = steady_state_fc

        # Compute MVR
        result = compute_mvr(inputs, current_year=datetime.now().year)
        st.success("✅ MVR Computed Successfully!")
        st.session_state["last_result"] = result

        # Display submodel predictions
        st.subheader("Submodel Predictions")
        st.write(f"**Company Name:** {st.session_state.get('company_name', 'N/A')}")
        st.write(f"**Normalized Capex (DFC):** {result['display']['Normalized Capex (DFC)']}")
        st.write(f"**Steady-State Fixed Costs (Total):** {result['display']['Steady-State Fixed Costs (Total)']}")
        st.write(f"**Steady-State Fixed Costs (Per Person):** {result['display']['Steady-State Fixed Costs (Per Person)']}")
        st.write(f"**Gross Margin (GM):** {result['display']['Gross Margin (GM)']}")

        # Display submodel errors
        st.subheader("Submodel Errors")
        sub_errors = result["submodel_errors"]
        st.write(f"**Normalized Capex (DFC) Error:** {sub_errors['Normalized Capex (DFC)']}")
        st.write(f"**Steady-State (FCP) Error:** {sub_errors['Steady-State (FCP)']}")
        st.write(f"**Gross Margin (GM) Error:** {sub_errors['Gross Margin (GM)']}")

        # Display final MVR results
        st.subheader("Estimated MVR")
        st.write(f"**MVR Value:** {result['mvr_value_display']}")
        st.write(f"**Confidence Interval (±):** {result['error_display']}")

        # Display MVR Equation and Explanation
        st.markdown("---")
        st.markdown(
            f"""
            $$\\text{{MVR}} = \\frac{{((\\text{{Fixed Costs per Person}} \\times \\text{{Headcount}}) \\times (1 + \\text{{Normalized Capex}}))}}{{\\text{{Gross Margin}}}}$$
            """
        )
        st.markdown(
            "The **Minimum Viable Revenue (MVR)** represents the lowest revenue level at which a company "
            "can sustain operations while covering fixed and variable costs, based on predicted financial drivers."
        )

# -----------------------------
# Section 3: Export Results Button(s)
# -----------------------------
if "last_result" in st.session_state:
    st.header("Export Results")
    df_result = pd.DataFrame([{
        "DFC_pred": st.session_state["last_result"]["dfc_pred"],
        "FCP_pred": st.session_state["last_result"]["fcp_pred"],
        "GM_pred": st.session_state["last_result"]["gm_pred"],
        "MVR": st.session_state["last_result"]["mvr_value"],
        "Error": st.session_state["last_result"]["error"]
    }])
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export as CSV", data=csv, file_name="MVR_Estimate.csv", mime="text/csv")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_result.to_excel(writer, index=False, sheet_name="MVR")
    st.download_button("📊 Export as Excel", data=buffer.getvalue(), file_name="MVR_Estimate.xlsx", mime="application/vnd.ms-excel")

# -----------------------------
# Section 4: Restart Option Button
# -----------------------------
st.divider()
if st.button("🔄 New Estimate"):
    keys_to_clear = [
        "use_file", "uploaded_file", "year_founded", "trading_status", "headcount",
        "segments", "has_financials", "gm_percent", "norm_capex",
        "steady_fc", "last_result", "data_loaded"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<small>Disclaimer: This tool provides estimates only. © 2025</small>", unsafe_allow_html=True)
