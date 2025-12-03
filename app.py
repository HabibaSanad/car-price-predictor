import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("car_price_model.pkl")

# Page title
st.title(" Car Price Prediction App")
st.write("""
Welcome to the Car Price Prediction App!

This tool uses a Machine Learning model to estimate car prices based on 
basic vehicle features such as year, age, mileage, and engine size.

Select the information below to get an instant prediction.
""")

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write("""
**Car Price Predictor **

- Built using **Python**, **Pandas**, **Scikit-Learn**, and **Streamlit**
- ML Model: **Random Forest Regressor**
- Developer: **Habiba Abdelmalik**
""")
st.sidebar.markdown("---")
st.sidebar.write("GitHub: https://github.com/HabibaSanad")

# Manufacturer list
manufacturers = ["BMW", "Ford", "Porsche", "Toyota", "VW"]

# User inputs
manufacturer = st.selectbox("Choose Manufacturer", manufacturers)
year = st.number_input("Year of manufacture", 1990, 2025, 2015)
car_age = st.number_input("Car age", 0, 35, 5)
engine_size = st.number_input("Engine size (L)", 1.0, 6.0, 2.0)
mileage = st.number_input("Mileage", 0, 500000, 50000)

if st.button("Predict Price"):
    # Create DataFrame with all model columns = 0
    input_df = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))),
                             columns=model.feature_names_in_)

    # Fill the known numeric values
    input_df["Year of manufacture"] = year
    input_df["Car age"] = car_age
    input_df["Engine size"] = engine_size
    input_df["Mileage"] = mileage

    # One-Hot encoding for manufacturer
    manu_col = f"Manufacturer_{manufacturer}"
    if manu_col in input_df.columns:
        input_df[manu_col] = 1

    # Predict
    price_pred = model.predict(input_df)[0]

    # Display
    st.markdown(f"""
    <div style="padding:15px; background:#dfffe2; border-radius:10px; border-left:7px solid #3cc46c;">
        <h2 style="color:#1e7e34;">Predicted Car Price: ${price_pred:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Developed by **Habiba Abdelmalik** © 2025")
