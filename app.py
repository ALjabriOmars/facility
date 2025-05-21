import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Facility Cost Estimator", layout="centered")
st.title("🏭 Facility Assignment Cost Estimator")
st.markdown("Estimate the **total cost** (in OMR) for assigning a facility to a location.")

# User inputs
space_required = st.slider("Space Required (units)", min_value=10, max_value=100, value=50)
location_capacity = st.slider("Location Capacity (units)", min_value=10, max_value=150, value=75)
assignment_cost = st.slider("Assignment Cost (OMR)", min_value=100, max_value=1000, value=500)
flow_distance_cost = st.slider("Flow-Distance Cost (OMR)", min_value=100000, max_value=500000, value=300000)

# Combine input
input_data = np.array([[space_required, location_capacity, assignment_cost, flow_distance_cost]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Estimate Cost"):
    predicted_cost = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated Total Cost: **{predicted_cost:,.2f} OMR**")
