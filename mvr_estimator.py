# -----------------------------
# Import Libraries
# -----------------------------

import streamlit as st
import pandas as pd
from io import BytesIO
from model_backend import compute_mvr, compute_mvr_batch
from datetime import datetime
from PIL import Image

# -----------------------------
# Section 1: Page Configuration
# -----------------------------

# Single-page layout with headers and description
st.set_page_config(page_title="MVR Estimator", layout="centered")

# Display the InsightOrca logo
logo=Image.open("InsightOrca-standard.png")
st.image(logo, width=200)

st.title("Minimum Viable Revenue (MVR) Estimation Tool")
st.markdown(
    "Estimate the **Minimum Viable Revenue (MVR)** to assess company and overall market health using financial and/or non-financial indicators. Please see the [Documentation](https://) for details."
)

# -----------------------------
# Section 2: Company Information Input
# -----------------------------

# Ask the user if they want to upload a file
use_file = st.radio("Load company data from a file? (Note: Please see documentation to ensure the source file has the required fields.)",
                    ("Yes", "No"),
                    horizontal=True,
                    key="use_file")

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

                # Check that expected column names exist
                expected = [
                    "Company Name", "Year Founded", "Headcount", "Trading Status", "Startup Status",
                    "Reseller/VAR", "Data Services", "Infrastructure Services", "Hardware", "Software",
                    "Asset-Intensive", "Labor Services", "Engineering & Advisory Services",
                    "Company Labor Class", "Satellite Owner"
                ]
                missing = [col for col in expected if col not in df.columns]
                if missing:
                    st.warning(f"⚠️ The following expected columns are missing: {', '.join(missing)}. Please ensure your file has the correct format as specified in the Documentation.")

                st.session_state["uploaded_df"] = df
                st.session_state["date_loaded"] = True

            except Exception as e:
                st.error(f"Error reading file: {e}. Please try again")

    # If the user does not want to upload a file:
    else:
        st.subheader("Step 1: Enter Company Information")
        col1, col2 = st.columns(2)

        # First column inputs
        with col1:
            company_name = st.text_input("Company Name (optional)", key="company_name")
            company_data["Company Name"] = company_name

            company_data["Headcount"] = st.number_input(
                "Headcount", min_value=1, step=1, value=10, key="headcount"
            )

            startup_input = st.selectbox("Is a Startup?", ["Yes", "No"], key="startup_status")
            company_data["is_startup"] = 1 if startup_input == "Yes" else 0

            labor_class = st.selectbox("Company Labor Class", ["A", "B", "C"], key="company_labor_class_input")
            company_data["company_labor_class"] = labor_class

        # Second column inputs
        with col2:
            company_data["Year Founded"] = st.number_input(
            "Year Founded", 1900, datetime.now().year, 2015, key="year_founded"
            )

            company_data["Trading Status"] = st.selectbox(
                "Trading Status", ["Public", "Private"], key="trading_status"
            )

            # Satellite ownership
            owns_sat = st.selectbox("Owns Satellites?", ["Yes", "No"], key="has_sat_input")
            company_data["has_sat"] = 1 if owns_sat == "Yes" else 0

            segments = st.multiselect(
                "Revenue Driver Attributes",
                [
                    "Asset-Intensive", "Engineering & Advisory", "Data Services",
                    "Infrastructure Services", "Labor", "Hardware", "Software", "Reseller/VAR"
                ],
                key="segments"
            )
            # Encode segments as 1/0
            segment_map = {
                "Asset-Intensive": "Asset-Intensive",
                "Engineering & Advisory": "Engineering_and_Advisory_Services",
                "Data Services": "Data_Services",
                "Infrastructure Services": "Infrastructure_Services",
                "Labor": "Labor_Services",
                "Hardware": "Hardware",
                "Software": "Software",
                "Reseller/VAR": "Reseller/VAR"
            }
            for seg_key in segment_map.values():
                company_data[seg_key] = 0
            for seg in segments:
                key_name = segment_map[seg]
                company_data[key_name] = 1

            # Company size medium
            headcount = company_data["Headcount"]
            company_data["company_size_medium"] = 1 if 50 <= headcount <= 500 else 0

        # Ensure mvr_startup_score_binary exists
        company_data["mvr_startup_score_binary"] = 0

        # Create financial information section
        st.subheader("Step 2: Enter Financial Information (optional)")
        st.markdown("Note that input must contain the correct units.")

        has_financials = st.radio(
            "Is Gross Margin, Normalized Capex, and/or Steady-State Fixed Costs known?",
            ("Yes", "No"),
            horizontal=True,
            key="has_financials"
        )

        if has_financials == "Yes":
            col1, col2 = st.columns(2)
            with col1:
                gm_percent = st.number_input("Gross Margin Percentage (e.g., 50%)", 0.0, 100.0, 0.0, 10.0, key="gm_percent")
                steady_state_fc = st.number_input("Steady-State Fixed Costs (in US$ thousands)", 0.0, step=1000.0, key="steady_fc") # millions more common  to avoid large numbers
           
            with col2:
                capex_known = st.radio(
                    "Is Normalized Capex known?",
                    options=("Yes", "No"),
                    index=1,
                    horizontal=True,
                    key="capex_known_radio"
                )
                if capex_known == "Yes":
                    norm_capex = st.number_input("Enter Normalized Capex (in US$ thousands):", min_value=0.0, step=1000.0, key="norm_capex_input")
                    if norm_capex <= 0:
                        st.warning("⚠️ Normalized Capex must be greater than zero.")
                else:
                    st.info("Normalized Capex will be estimated using the submodel estimates.")
                    norm_capex = None
        else:
            gm_percent = norm_capex = steady_state_fc = None

    # Ensure "Estimate MVR" button is at the bottom of the form
    submitted = st.form_submit_button("Estimate MVR")

    if submitted:
        if use_file == "Yes" and "uploaded_df" in st.session_state:
            df_input = st.session_state["uploaded_df"]
            results_list = compute_mvr_batch(df_input, current_year=datetime.now().year)

            # Extract only user-friendly fields
            friendly_results = []
            for i, r in enumerate(results_list):
                friendly_results.append({
                    "Company Name": df_input.loc[i, "Company Name"],  # original file name
                    "Normalized Capex (DFC) ($000s)": r["display"]["Normalized Capex (DFC)"],
                    "Steady-State Fixed Costs (Total) ($000s)": r["display"]["Steady-State Fixed Costs (Total)"],
                    "Steady-State Fixed Costs (Per Person) ($000s)": r["display"]["Steady-State Fixed Costs (Per Person)"],
                    "Gross Margin (%)": r["display"]["Gross Margin (GM)"],
                    "Estimated MVR ($000s)": r["mvr_value_display"],
                    "Confidence Interval ($000s)": r["error_display"]
                })

            df_result = pd.DataFrame(friendly_results)
            st.session_state["last_result"] = df_result
            st.dataframe(df_result)
            st.success("✅ MVR Computed Successfully!")
        
        else:
            inputs = company_data.copy()
            if gm_percent and gm_percent > 0:
                inputs["Gross Margin %"] = gm_percent
            if norm_capex and norm_capex > 0:
                inputs["Normalized Capex"] = norm_capex
            if steady_state_fc and steady_state_fc > 0:
                inputs["Steady-State Fixed Costs"] = steady_state_fc

            result = compute_mvr(inputs, current_year=datetime.now().year)
            st.success("✅ MVR Computed Successfully!")
            st.session_state["last_result"] = result

            # Display submodel predictions
            st.subheader("Submodel Predictions")
            st.write(f"**Company Name:** {st.session_state.get('company_name', 'N/A')}")
            st.write(f"**Normalized Capex (D) (in US thousands):** {result['display']['Normalized Capex (DFC)']}")
            st.write(f"**Steady-State Fixed Costs (Total) (in US thousands):** {result['display']['Steady-State Fixed Costs (Total)']}")
            st.write(f"**Steady-State Fixed Costs (Per Person) (in US thousands):** {result['display']['Steady-State Fixed Costs (Per Person)']}")
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
                $$\\text{{MVR}} = \\frac{{((\\text{{FCP}} \\times \\text{{Headcount}}) \\times (1 + \\text{{D/FC}}))}}{{\\text{{GM}}}}$$
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
    df_result = st.session_state["last_result"]

    if isinstance(df_result, pd.DataFrame):
        # Ensure the company_name column exists and is used
        if "company_name" in df_result.columns:
            export_df = df_result.copy()
        else:
            # fallback: create a company_name column with placeholders
            export_df = df_result.copy()
            export_df["company_name"] = [f"Company {i+1}" for i in range(len(df_result))]

        # Reorder columns to have Company Name first
        cols = export_df.columns.tolist()
        if "company_name" in cols:
            cols.insert(0, cols.pop(cols.index("company_name")))
        export_df = export_df[cols]

        # Export CSV
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export as CSV", data=csv, file_name="MVR_Estimate.csv", mime="text/csv")

        # Export Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, index=False, sheet_name="MVR")
        st.download_button("📊 Export as Excel", data=buffer.getvalue(), file_name="MVR_Estimate.xlsx", mime="application/vnd.ms-excel")

    else:
        # Single company (dict output)
        single_result = df_result.copy() if isinstance(df_result, dict) else dict(df_result)
        single_result["company_name"] = single_result.get("company_name", st.session_state.get("company_name", "N/A"))

        csv = pd.DataFrame([single_result]).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export as CSV", data=csv, file_name="MVR_Estimate.csv", mime="text/csv")

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            pd.DataFrame([single_result]).to_excel(writer, index=False, sheet_name="MVR")
        st.download_button("📊 Export as Excel", data=buffer.getvalue(), file_name="MVR_Estimate.xlsx", mime="application/vnd.ms-excel")


# -----------------------------
# Section 4: Restart Option Button
# -----------------------------
st.divider()
if st.button("🔄 New Estimate"):
    keys_to_clear = [
        "use_file", "uploaded_file", "year_founded", "trading_status", "headcount",
        "segments", "has_financials", "gm_percent",
        "norm_capex_input", "capex_known_radio",
        "steady_fc", "last_result", "data_loaded", "uploaded_df"
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