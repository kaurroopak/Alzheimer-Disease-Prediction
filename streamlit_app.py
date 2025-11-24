import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
from PIL import Image
import numpy as np
import tensorflow as tf

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Alzheimer's Risk Prediction", layout="wide")

# ------------------------------------------------------
# LOAD TABULAR ML MODEL
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "XGBmodel.pkl")
model = joblib.load(model_path)

# ------------------------------------------------------
# LOAD MRI TFLITE MODEL
# ------------------------------------------------------
tflite_path = os.path.join(BASE_DIR, "vgg16_binary_fp16.tflite")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_mri(image):
    img = image.convert("RGB").resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    result = interpreter.get_tensor(output_details[0]['index'])
    return float(result[0][0])   # Demented probability


# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.title("🧠 Alzheimer's Disease Risk Prediction System")
st.write("A unified tool for predicting dementia risk using **clinical features** and **MRI scan analysis**.")

st.markdown("---")

# ------------------------------------------------------
# SIDEBAR INFO
# ------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Feature Information")
    st.info("""
### Demographic  
• **Age** — Patient age  
• **Gender** — M/F  

### Education & SES  
• **Educ** — Years of education  
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
# TABULAR ML SECTION
# ------------------------------------------------------
st.header("📋 Clinical Feature-Based Risk Prediction")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        patient_name = st.text_input("Patient Name (Optional)")
        gender = st.selectbox(
            "Gender",
            ['M', 'F'],
            help="Recorded during clinical visit. Source: Patient demographics."
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

    submit_btn = st.form_submit_button("🔍 Predict Using Clinical Features")

# ------------------------------------------------------
# TABULAR PREDICTION LOGIC
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

    st.subheader("📊 Prediction Results")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Non-Demented Probability", f"{non_demented_prob*100:.2f}%")
    with c2:
        st.metric("Demented Probability", f"{demented_prob*100:.2f}%")

    if demented_prob > 0.30:
        risk_level = "High Risk (Likely Demented)"
        st.error("⚠️ High Risk — Further clinical evaluation recommended.")
    else:
        risk_level = "Low Risk (Likely Non-Demented)"
        st.success("✅ Low Risk — Likely Non-Demented.")

    st.progress(min(demented_prob, 1.0))

    # ------------------------------------------------------
    # PDF GENERATION
    # ------------------------------------------------------
    def generate_pdf(name, input_df, dp, ndp, risk):

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "Alzheimer's Disease Prediction Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", size=10)

        if name.strip():
            pdf.cell(200, 8, f"Patient Name: {name}", ln=True)

        pdf.cell(200, 10, "Patient Information:", ln=True)
        for col in input_df.columns:
            pdf.cell(200, 8, f"{col}: {input_df[col].values[0]}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, "Prediction Results:", ln=True)
        pdf.cell(200, 8, f"Demented Probability: {dp*100:.2f}%", ln=True)
        pdf.cell(200, 8, f"Non-Demented Probability: {ndp*100:.2f}%", ln=True)
        pdf.cell(200, 8, f"Risk Level: {risk}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", "I", size=9)
        pdf.multi_cell(
            0, 6,
            "Disclaimer: This report is generated using a machine learning model. "
            "It is NOT a medical diagnosis. Consult a licensed neurologist for clinical interpretation."
        )

        output_file = "alzheimers_report.pdf"
        pdf.output(output_file)
        return output_file

    pdf_file = generate_pdf(patient_name, input_data, demented_prob, non_demented_prob, risk_level)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Patient Report (PDF)",
            data=f,
            file_name="Alzheimers_Report.pdf",
            mime="application/pdf"
        )


# ------------------------------------------------------
# MRI VGG16 SCAN ANALYSIS
# ------------------------------------------------------
st.markdown("---")
st.header("🧠 MRI Scan Analysis (Deep Learning)")

uploaded_img = st.file_uploader("Upload MRI Scan (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img = Image.open(uploaded_img)
    st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("Analyze MRI Scan"):
        mri_prob = predict_mri(img)
        mri_non_prob = 1 - mri_prob

        st.subheader("📊 MRI Scan Prediction")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Non-Demented Probability", f"{mri_non_prob*100:.2f}%")
        with c2:
            st.metric("Demented Probability", f"{mri_prob*100:.2f}%")

        if mri_prob > 0.30:
            st.error("⚠️ High Risk — MRI indicates dementia-like patterns.")
        else:
            st.success("✅ Low Risk — MRI does not show dementia patterns.")

        st.progress(min(mri_prob, 1.0))


# ------------------------------------------------------
# FINAL DISCLAIMER
# ------------------------------------------------------
st.markdown("### ⚠️ Medical Disclaimer")
st.warning("""
This tool provides **risk estimation**, not a clinical diagnosis.  
Always consult a **licensed neurologist** for medical decisions.
""")
