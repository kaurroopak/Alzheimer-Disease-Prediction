import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "XGBmodel.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Alzheimer's Prediction", layout="wide")

# ------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------
st.title("🧠 Alzheimer's Disease Risk Prediction")
st.write("Predict dementia risk using MRI-based features and cognitive test scores.")

st.markdown("---")

# ------------------------------------------------------
# SIDEBAR FEATURE GUIDE
# ------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the Features")
    st.info("""
### Demographic  
• **Age** — Patient age  
• **Gender** — M/F  

### Education & SES  
• **Educ** — Years of formal education  
• **SES** — Socioeconomic Status (1–5)

### Cognitive Assessment  
• **MMSE** — Mini-Mental State Exam (0–30)  
  Lower score → Higher impairment risk

### MRI Biomarkers  
• **eTIV** — Total intracranial volume  
• **nWBV** — Normalized whole brain volume  
• **ASF** — Atlas scaling factor  
""")

# ------------------------------------------------------
# PATIENT INPUT FORM
# ------------------------------------------------------
st.subheader("Enter Patient Details")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        patient_name = st.text_input(
            "Patient Name (Optional)",
            "",
            help="Only used in PDF report. You may leave it empty."
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            help="Patient biological sex recorded during clinical intake."
        )

        age = st.number_input(
            "Age",
            min_value=50,
            max_value=100,
            value=70,
            help="Patients in OASIS dataset range from 50–100 years."
        )

    # Column 2
    with col2:
        educ = st.number_input(
            "Years of Education (Educ)",
            min_value=0,
            max_value=25,
            value=12,
            help="Years of formal education completed."
        )

        ses = st.number_input(
            "Socioeconomic Status (SES)",
            min_value=1,
            max_value=5,
            value=2,
            help="Clinician-rated SES score (1 = Low, 5 = High)."
        )

        mmse = st.number_input(
            "MMSE Score",
            min_value=0,
            max_value=30,
            value=28,
            help="Cognitive screening score (0–30). Below 24 indicates impairment."
        )

    # Column 3
    with col3:
        etiv = st.number_input(
            "eTIV (MRI)",
            min_value=1000,
            max_value=2000,
            value=1500,
            help="Estimated total intracranial volume from MRI."
        )

        nwbv = st.number_input(
            "nWBV (MRI)",
            min_value=0.60,
            max_value=0.90,
            value=0.70,
            step=0.01,
            help="Normalized whole brain volume from MRI segmentation."
        )

        asf = st.number_input(
            "ASF (MRI)",
            min_value=1.0,
            max_value=2.0,
            value=1.5,
            step=0.01,
            help="Atlas scaling factor used during MRI registration."
        )

    submit_btn = st.form_submit_button("🔍 Predict Risk")

# ------------------------------------------------------
# PREDICTION LOGIC
# ------------------------------------------------------
if submit_btn:

    gender_numeric = 1 if gender == "M" else 0

    input_data = pd.DataFrame({
        "M/F": [gender_numeric],
        "Age": [age],
        "Educ": [educ],
        "SES": [ses],
        "MMSE": [mmse],
        "eTIV": [etiv],
        "nWBV": [nwbv],
        "ASF": [asf]
    })

    demented_prob = float(model.predict_proba(input_data)[0][1])
    non_demented_prob = 1 - demented_prob

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # Probability metrics
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Non-Demented Probability", f"{non_demented_prob*100:.2f}%")
    with c2:
        st.metric("Demented Probability", f"{demented_prob*100:.2f}%")

    # Risk classification
    if demented_prob > 0.30:
        risk_level = "High Risk (Likely Demented)"
        st.error("⚠️ High Risk — Further clinical evaluation recommended.")
    else:
        risk_level = "Low Risk (Likely Non-Demented)"
        st.success("✅ Low Risk — Likely Non-Demented.")

    st.progress(min(demented_prob, 1.0))

    # ------------------------------------------------------
    # PDF REPORT GENERATION
    # ------------------------------------------------------

    def generate_pdf(name, input_data, dp, ndp, risk):

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", size=10)

        # Patient Name
        if name.strip() != "":
            pdf.cell(200, 8, txt=f"Patient Name: {name}", ln=True)

        pdf.cell(200, 10, txt="Patient Information:", ln=True)
        for col in input_data.columns:
            pdf.cell(200, 8, txt=f"{col}: {input_data[col].values[0]}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, txt="Prediction Results:", ln=True)
        pdf.cell(200, 8, txt=f"Demented Probability: {dp*100:.2f}%", ln=True)
        pdf.cell(200, 8, txt=f"Non-Demented Probability: {ndp*100:.2f}%", ln=True)
        pdf.cell(200, 8, txt=f"Risk Level: {risk}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "I", size=9)
        pdf.multi_cell(
            0, 6,
            "Disclaimer: This report is generated using a machine learning model "
            "trained on MRI-based features. This is NOT a medical diagnosis. "
            "Always consult a licensed healthcare professional."
        )

        output = "alzheimers_report.pdf"
        pdf.output(output)
        return output

    # Create & download PDF
    pdf_file = generate_pdf(patient_name, input_data, demented_prob, non_demented_prob, risk_level)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Patient Report (PDF)",
            data=f,
            file_name="Alzheimers_Report.pdf",
            mime="application/pdf"
        )

# ------------------------------------------------------
# DISCLAIMER
# ------------------------------------------------------
st.markdown("### ⚠️ Important Disclaimer")
st.warning("""
This tool provides a **risk estimation**, not a clinical diagnosis.  
Always consult a **licensed neurologist** for medical decisions.
""")
