import streamlit as st
import numpy as np
import pickle

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Real Estate AI Valuation Hub",
    page_icon="🏠",
    layout="centered"
)

# 2. Premium Real Estate Theme (CSS Background Graphic Injection)
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.95)), 
                          url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f8fafc;
    }
    .stNumberInput div div input {
        background-color: rgba(30, 41, 59, 0.9) !important;
        color: #fbbf24 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px;
        font-weight: bold;
    }
    label p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #d97706 0%, #b45309 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 14px 20px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(180, 83, 94, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 0 25px #f59e0b;
        color: #0f172a;
    }
    .valuation-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 30px;
        background-color: rgba(30, 41, 59, 0.8);
        border: 2px solid #fbbf24;
        color: #fbbf24;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 Real Estate Market AI Valuation Engine")
st.markdown("### Automated Predictive Appraiser using XGBoost Regression")
st.write("Provide the 13 required structural and neighborhood characteristics metrics below.")
st.markdown("---")

# 3. Load Saved Model Artifact
try:
    with open('house_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
except FileNotFoundError:
    st.error("🚨 File Loss Error: 'house_model.pkl' could not be found. Please run Step 1 first.")
    st.stop()

st.markdown("#### Property Specifications Profile (13 Features)")

# Organizing 13 inputs into 3 balanced column sections
col1, col2, col3 = st.columns(3)

with col1:
    crim = st.number_input("Per Capita Crime Rate (CRIM)", value=0.00632, format="%.5f")
    zn = st.number_input("Residential Land Zones % (ZN)", value=18.0)
    indus = st.number_input("Non-retail Business Acres % (INDUS)", value=2.31)
    chas = st.number_input("Tract Bounds Charles River (CHAS: 0 or 1)", min_value=0, max_value=1, value=0)
    nox = st.number_input("Nitric Oxides Concentration (NOX)", value=0.538, format="%.3f")

with col2:
    rm = st.number_input("Average Rooms per Dwelling (RM)", value=6.575)
    age = st.number_input("Units Built before 1940 % (AGE)", value=65.2)
    dis = st.number_input("Weighted Distances to Employment (DIS)", value=4.090)
    rad = st.number_input("Accessibility Index to Highways (RAD)", value=1.0)

with col3:
    tax = st.number_input("Property Tax Rate per $10k (TAX)", value=296.0)
    ptratio = st.number_input("Pupil-Teacher Ratio (PTRATIO)", value=15.3)
    b = st.number_input("Proportion of Minorities Metric (B)", value=396.9)
    lstat = st.number_input("Lower Status of Population % (LSTAT)", value=4.98)

# 4. Prediction Execution
if st.button("CALCULATE MARKET ESTIMATE"):
    # Group all 13 variables in the exact array sequence expected by XGBoost
    features = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]
    
    # Reshape features to 1 row, 13 columns array
    input_array = np.asarray(features).reshape(1, -1)
    
    # Run prediction
    prediction = model.predict(input_array)
    
    # Format outcome scale display safely
    estimated_price = prediction[0] * 10000 if prediction[0] > 0 else 0
    
    st.markdown(
        f'<div class="valuation-box">'
        f'🏷️ ESTIMATED PROPERTY VALUATION:<br>'
        f'<span style="font-size: 36px; color: #ffffff;">${estimated_price:,.2f}</span>'
        f'</div>', 
        unsafe_allow_html=True
    )
