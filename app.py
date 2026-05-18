
import streamlit as st
import pandas as pd
import joblib

st.title("PCOS Prediction System")

model = joblib.load('models/hybrid_model.pkl')

st.header("Enter Patient Details")

age = st.number_input("Age", 15, 50)
weight = st.number_input("Weight (Kg)", 30.0, 120.0)
height = st.number_input("Height (Cm)", 120.0, 200.0)
bmi = st.number_input("BMI", 10.0, 50.0)

if st.button("Predict PCOS"):
    sample = pd.DataFrame([[1,1,0,age,weight,height,bmi] + [0]*37])

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("PCOS Detected")
    else:
        st.success("No PCOS Detected")
