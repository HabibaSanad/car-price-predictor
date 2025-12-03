import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")

# Manufacturer options
manufacturers = ["BMW", "Ford", "Porsche", "Toyota", "VW"]

st.title("Car Price Prediction 🚗")

# User inputs
manufacturer = st.selectbox("Choose Manufacturer", manufacturers)
year = st.number_input("Year of manufacture", 1990, 2025, 2015)
car_age = st.number_input("Car age", 0, 35, 5)
engine_size = st.number_input("Engine size (L)", 1.0, 6.0, 2.0)
mileage = st.number_input("Mileage", 0, 500000, 50000)

if st.button("Predict Price"):
    # Create DataFrame with zeros for all model features
    input_df = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)

    # Fill user input values
    input_df["Year of manufacture"] = year
    input_df["Car age"] = car_age
    input_df["Engine size"] = engine_size
    input_df["Mileage"] = mileage

    # Set manufacturer column
    manufacturer_col = f"Manufacturer_{manufacturer}"
    if manufacturer_col in input_df.columns:
        input_df[manufacturer_col] = 1

    # Predict price
    price_pred = model.predict(input_df)[0]
    st.success(f"Predicted Car Price: ${price_pred:,.2f}")
