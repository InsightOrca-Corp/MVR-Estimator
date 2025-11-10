# -----------------------------
# Import Libraries
# -----------------------------

import streamlit as st
import pandas as pd
from io import BytesIO
from model_backend import compute_mvr, compute_mvr_batch
from datetime import datetime
from PIL import Image
import logging
import traceback

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    filename="mvr_estimator_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Section 1: Page Configuration
# -----------------------------
def main():
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
                        index=1,
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
                    df = pd.read_csv(uploaded_file, index_col=None) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, index_col=None)
                    df.columns = df.columns.str.strip()

                    # Check that expected column names exist
                    expected = [
                        "Company Name", "Year Founded", "Headcount", "Trading Status",
                        "Reseller/VAR", "Data Services", "Infrastructure Services", "Hardware", "Software",
                        "Asset-Intensive", "Labor Services", "Engineering & Advisory Services",
                        "Company Labor Class", "Satellite Owner"
                    ]
                    missing = [col for col in expected if col not in df.columns]

                    if missing:
                        st.error(
                            "❌ Uploaded file does not conform to the required schema. "
                            "Missing columns: " + ", ".join(missing))
                        
                        # Show a "Try Again" button if user uploads invalid file
                        if st.form_submit_button("Try Again"):
                            file_clears = ["uploaded_file", "data_loaded", "uploaded_df"]
                            for k in file_clears:
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()
                    else:
                        # File is valid
                        st.session_state["uploaded_df"] = df
                        st.session_state["date_loaded"] = True
                        st.success("✅ File uploaded successfully with all required columns.")
                        st.dataframe(df.head())

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

                labor_class = st.selectbox("Company Labor Class", ["A", "B", "C", "D", "E"], key="company_labor_class_input")
                company_data["company_labor_class"] = labor_class

                segments = st.multiselect(
                    "Revenue Driver Attributes",
                    [
                        "Asset-Intensive", "Engineering & Advisory", "Data Services",
                        "Infrastructure Services", "Labor", "Hardware", "Software", "Reseller/VAR"
                    ],
                    key="segments"
                )

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
                    gm_percent = st.number_input("Gross Margin Percentage (e.g., 50%)", 0.0, 100.0, 0.0, 0.1, key="gm_percent")
                    steady_state_fc = st.number_input("Steady-State Fixed Costs (in US$ thousands)", 0.0, step=1000.0, key="steady_fc") # millions more common  to avoid large numbers
            
                with col2:
                    norm_capex = st.number_input(
                        "Normalized Capex (in US$ thousands)", 0.0, step=1000.0, key="norm_capex_input"
                    )
                    if norm_capex <= 0:
                        st.warning("⚠️ Normalized Capex must be greater than zero. Otherwise, submodel estimates will be used.")
            else:
                gm_percent = norm_capex = steady_state_fc = None

        # Ensure "Estimate MVR" button is at the bottom of the form
        submitted = st.form_submit_button("Estimate MVR")

        if submitted:
            if use_file == "Yes":
                if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
                    df_input = st.session_state["uploaded_df"]
                    results_list = compute_mvr_batch(df_input, current_year=datetime.now().year)

                    # Extract only user-friendly fields
                    friendly_results = []
                    for i, r in enumerate(results_list):
                        row = {}
                        for col_name, display_key in [
                            ("Company Name", "Company Name"),
                            ("Normalized Capex (Total) ($000s)", "Normalized Capex (Total)"),
                            ("Steady-State Fixed Costs (Total) ($000s)", "Steady-State Fixed Costs (Total)"),
                            ("Steady-State Fixed Costs (Per Person) ($000s)", "Steady-State Fixed Costs (Per Person)"),
                            ("Gross Margin (%)", "Gross Margin (GM)"),
                            ("Estimated MVR", "mvr_value_display")
                        ]:
                            if col_name == "Company Name":
                                row[col_name] = df_input.loc[i, col_name] if col_name in df_input.columns else "N/A"
                            elif display_key != "mvr_value_display":
                                row[col_name] = r["display"].get(display_key, "N/A")
                            else:
                                row[col_name] = r.get(display_key, "N/A")
                        friendly_results.append(row)

                    if friendly_results:
                        results_df = pd.DataFrame(friendly_results)
                        st.session_state["last_result"] = results_df
                        st.success("✅ MVR estimates computed successfully for uploaded file!")
                        st.dataframe(results_df)
                    else:
                        st.warning("⚠️ No results to display. Please check if your input columns match the required schema.")
            
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
                st.write(f"**Normalized Capex (Total) (in US thousands):** {result['display']['Normalized Capex (Total)']}")
                st.write(f"**Steady-State Fixed Costs (Total) (in US thousands):** {result['display']['Steady-State Fixed Costs (Total)']}")
                st.write(f"**Steady-State Fixed Costs (Per Person) (in US thousands):** {result['display']['Steady-State Fixed Costs (Per Person)']}")
                st.write(f"**Gross Margin (GM):** {result['display']['Gross Margin (GM)']}")

                # Display final MVR results
                st.subheader("Estimated MVR")
                st.write(f"**MVR Value:** {result['mvr_value_display']}")

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
            "norm_capex_input",
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

# -----------------------------
# Global Exception Handling
# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("⚠️ An unexpected error occurred. Please refresh the page and try again.")
        logging.error(traceback.format_exc())