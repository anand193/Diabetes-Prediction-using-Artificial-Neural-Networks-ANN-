import streamlit as st
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model

# Load saved model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = load_model('model.h5', compile=False)

# Streamlit app
st.title("🧠 Diabetes Prediction using ANN")
st.write("This app predicts the likelihood of diabetes based on medical inputs.")

# Collect user inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=130, value=70)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prepare and scale input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, diabetes_pedigree, age]])

input_data_scaled = scaler.transform(input_data)

# Predict when user clicks the button
if st.button("Predict"):
    st.write("Input data shape:", input_data.shape)
    st.write("Scaled input:", input_data_scaled)

    if input_data_scaled is None:
        st.error("❌ input_data_scaled is None — check your scaler.pkl file!")
    else:
        prediction = model.predict(input_data_scaled)
        st.write("Raw model output:", prediction)
        predicted_class = (prediction > 0.5).astype(int)[0][0]

        if predicted_class == 1:
            st.error("⚠️ The model predicts a high likelihood of Diabetes.")
        else:
            st.success("✅ The model predicts a low likelihood of Diabetes.")
