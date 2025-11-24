# 🧠 Alzheimer's Disease Prediction using Machine Learning & Deep Learning Models

A comprehensive **Streamlit web application** that predicts **Alzheimer’s disease risk** using both:

- **Machine Learning (XGBoost)**  
- **Deep Learning (TensorFlow/Keras)** — optional MRI image-based model  
- **Tabular MRI biomarkers + cognitive test scores**

The app also generates a downloadable **PDF clinical-style report** for each prediction.

---

## 📌 Project Overview
This project aims to estimate the likelihood of dementia based on **MRI-derived structural brain features** and **cognitive assessment scores**.  
It uses :

- **8 clinically validated biomarkers** from the **OASIS Cross-Sectional Dataset**
- **MRI image data** from the **Alzheimer MRI 4 Classes Dataset**


## 🚀 Live Demo
[🔗 Alzheimer’s Risk Prediction App](https://alzheimer-disease-risk-prediction.streamlit.app/)

### 📂 Datasets Used

#### **1. OASIS-1 Cross-Sectional Dataset (Tabular + MRI Biomarkers)**  
Used for training the **XGBoost tabular model**.  
🔗 https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers

#### **2. Alzheimer MRI – 4 Class Brain MRI Dataset**  
Used for training the **VGG16 deep learning model** (later converted to binary: Demented vs Non-Demented).  
🔗 https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset

---

## 🛠️ Technologies Used

### **Machine Learning**
- XGBoost  
- Scikit-Learn  
- Pandas  
- NumPy  

### **Deep Learning**
- TensorFlow / Keras  
- TFLite Runtime (for lightweight inference)

### **Web Application**
- Streamlit  
- FPDF (for PDF Report Generation)

---

## 📂 Repository Structure
Alzheimer-Disease-Prediction/


├── Alzheimer'sDiseasePrediction.ipynb


├── alzheimers_prediction_(VGG16).ipynb


├── XGBmodel.pkl


├── vgg16_binary_fp16.tflite


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

