
# regression_app.py

import streamlit as st
import numpy as np
import pickle

# Load the model
with open("regression_model.pkl", 'rb') as f:
    model = pickle.load(f)

st.title("Simple Linear Regression Predictor")
st.write("This app uses a regression model trained on: y = 3x + 2")

# User input
x_input = st.number_input("Enter a value for x:", value=0)

# Prediction
if st.button("Predict"):
    x_array = np.array([[x_input]])
    y_pred = model.predict(x_array)
    st.success(f"Predicted y value: {y_pred[0]:.2f}")
