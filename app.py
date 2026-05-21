import streamlit as st
import requests

st.title("🏥 Hospital Diabetes Risk Dashboard")

st.subheader("Patient Input")

glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Analyze Patient"):

    payload = {
        "glucose": glucose,
        "bp": bp,
        "skin": skin,
        "insulin": insulin,
        "bmi": bmi,
        "dpf": dpf,
        "age": age
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()

    st.subheader("Risk Result")

    st.metric("Risk Score", f"{result['risk_score']:.2f}%")

    st.write("Risk Level:", result["risk_level"])
