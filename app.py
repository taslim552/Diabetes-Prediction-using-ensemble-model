import streamlit as st
import numpy as np
import joblib

# Load the trained scaler and stacked model
scaler = joblib.load("scaler.pkl")
meta_classifier = joblib.load("stacked_model.pkl")

# Title
st.title("🧠 Diabetes Prediction App (Stacked Ensemble)")
st.write("Enter the following details to predict diabetes likelihood:")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=26.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=2.5, value=0.500, step=0.001, format="%.5f")

age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Prediction button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = meta_classifier.predict(input_scaled)[0]
    probability = meta_classifier.predict_proba(input_scaled)[0]

    # Show result
    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.write(f"**Confidence:** {round(max(probability) * 100, 2)}%")
