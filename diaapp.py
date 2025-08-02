import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction App")

# Input features (accept all 8)
preg = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
bp = st.number_input('Blood Pressure')
skin = st.number_input('Skin Thickness')  # This will be dropped later
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

if st.button("Predict"):
    # 1. Prepare input with all 8 features
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # 2. Scale input
    input_scaled = scaler.transform(input_data)

    # 3. Remove SkinThickness (index 3)
    input_final = np.delete(input_scaled, 3, axis=1)  # Now only 7 features

    # 4. Predict
    prediction = model.predict(input_final)

    # 5. Show result
    if prediction[0] == 1:
        st.error("You are Diabetic 😢")
    else:
        st.success("You are Not Diabetic 😊")
