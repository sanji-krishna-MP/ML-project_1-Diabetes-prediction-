import streamlit as st
import pickle
import numpy as np


# Load model and scaler
model = pickle.load(open(r'D:\Python projects cv\Deployment of project p1(diabetes prediction)\model.pkl', 'rb'))
scaler = pickle.load(open(r'D:\Python projects cv\Deployment of project p1(diabetes prediction)\scaler.pkl', 'rb'))

st.title("Diabetes Prediction App")

# Input features
preg = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
bp = st.number_input('Blood Pressure')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("You are Diabetic 😢")
    else:
        st.success("You are Not Diabetic 😊")
