import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ---------------- Load Model, Scaler, Features ----------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_artifacts()

# ---------------- Title ----------------
st.title("🩺 Diabetes Prediction App")
st.write(
    "Enter the patient's health details below to predict the likelihood "
    "of diabetes using a Logistic Regression model."
)

st.divider()

# ---------------- Input Form ----------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

st.divider()

# ---------------- Prediction ----------------
if st.button("Predict", type="primary", use_container_width=True):
    # Build input dataframe in the exact order the model expects
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    input_df = pd.DataFrame([input_dict])[features]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High risk of Diabetes  \n**Probability: {probability * 100:.2f}%**")
    else:
        st.success(f"✅ Low risk of Diabetes  \n**Probability: {probability * 100:.2f}%**")

    st.progress(min(int(probability * 100), 100))

    with st.expander("See input summary"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")