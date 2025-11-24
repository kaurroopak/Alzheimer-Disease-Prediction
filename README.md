# 🧠 Alzheimer's Disease Prediction using Machine Learning Models

A Machine Learning-based Streamlit web app that predicts **Alzheimer’s disease risk** using MRI-derived brain measurements and cognitive test scores.  
The model is trained using **XGBoost**, achieving reliable classification between **Demented** and **Non-Demented** subjects.

---

## 📌 Project Overview
This project uses **tabular clinical + MRI features** to estimate the likelihood of dementia.  
The model takes 8 clinically relevant features:

## 🚀 Live Demo
[🔗 Alzheimer’s Risk Prediction App](https://alzheimer-disease-risk-prediction.streamlit.app/)

### **🧩 Dataset **
Using Oasis-Cross Sectional Dataset which can be found at -
https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers

---

## 🛠️ Technologies Used

### **Machine Learning**
- XGBoost  
- Scikit-Learn  
- Pandas  
- NumPy  

### **Web App**
- Streamlit  
- Joblib (for loading model)

---

## 📂 Repository Structure
Alzheimer-ML-Prediction/

│

├── Alzheimer'sDiseasePrediction.ipynb

├── oasis_cross-sectional.csv (Dataset)

├── XGBmodel.pkl

├── streamlit_app.py

├── requirements.txt

└── README.md

---

## ▶️ Run the Project Locally

### **1. Clone the Repository**

### **2. Install Dependencies**

    pip install -r requirements.txt

### **3. Run Streamlit App**

    streamlit run streamlit_app.py

---

## ⚠️ Important Disclaimer
This tool is NOT a medical diagnostic system.
It is a research/educational ML project and must not be used for clinical decisions.
