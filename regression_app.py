import streamlit as st
import numpy as np
import pickle
import os

MODEL_PATH = "regression_model.pkl"

st.title("Simple Linear Regression Predictor")
st.write("This app uses a regression model trained on: y = 3x + 2")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please make sure it's in the repo.")
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # User input
    x_input = st.number_input("Enter a value for x:", value=0)

    if st.button("Predict"):
        x_array = np.array([[x_input]])
        y_pred = model.predict(x_array)
        st.success(f"Predicted y value: {y_pred[0]:.2f}")
