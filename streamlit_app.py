import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("diabetes_xgb_model.joblib")

st.title("Diabetes Risk Prediction")
st.write("Enter your health details:")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict"):
    input_features = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0][1]

    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    probability_percent = prediction_proba * 100

    st.write("---")
    st.write(f"Predicted Class: {prediction} ({risk_level})")
    st.write(f"Probability of Diabetes: {probability_percent:.2f}%")
    
    if prediction == 1:
        st.error(f"⚠️ High Risk! You might have diabetes.")
        st.info("Recommendation: Consult a doctor for proper diagnosis and lifestyle guidance.")
    else:
        st.success(f"✅ Low Risk! Probability of diabetes is low.")
        st.info("Recommendation: Maintain a healthy lifestyle and monitor regularly.")
