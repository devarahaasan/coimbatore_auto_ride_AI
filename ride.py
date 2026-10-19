import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

# 1. SET UP STREAMLIT PAGE INTERFACE
st.set_page_config(page_title="Coimbatore RideFlow AI", page_icon="🛺", layout="wide")

# 2. LOAD ALL PRE-TRAINED MODELS WITH ERROR-PROOF DESERIALISATION
@st.cache_resource
def load_models():
    # Module 1 Models (Machine Learning)
    surge_model = joblib.load('Auto_ride_reg.pkl')
    cancel_model = joblib.load('Auto_ride_classi.pkl')

    # Module 2 Models (Deep Learning) - Safe Loading with compile=False to bypass Keras subclass mismatches
    eta_model = tf.keras.models.load_model('auto_dl.h5', compile=False)
    # Manually compile with simple strings to allow native runtime execution safely
    eta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    dl_scaler = joblib.load('auto_dl_scal.pkl')
    
  
    return surge_model, cancel_model, eta_model, dl_scaler

# Handle loading gracefully to prevent app crashes if files are missing in path
try:
    lr_model, rf_clf, lstm_model, scaler = load_models()
    model_status = "🟢 ALL MODELS LOADED SUCCESSFULLY"
except Exception as e:
    model_status = f"🔴 ERROR LOADING MODELS: {str(e)}"
    # Create mock fallbacks so the UI elements can still render without throwing NameErrors
    lr_model, rf_clf, lstm_model, scaler = None, None, None, None

# 3. SIDEBAR CONFIGURATION - LIVE SYSTEM OPERATIONS PANEL
st.sidebar.markdown(f"### **System Status:**\n`{model_status}`")
st.sidebar.header("🛺 Dispatch Control Panel")

# Dropdown list of core Coimbatore operational zones
zones = ['Saravanampatti', 'Gandhipuram', 'Peelamedu', 'RS Puram', 'Ukkadam']
pickup = st.sidebar.selectbox("Select Pickup Zone", zones, index=0)
drop = st.sidebar.selectbox("Select Drop Zone", zones, index=1)

# Dynamic environment sliders
distance = st.sidebar.slider("Estimated Route Distance (km)", 1.0, 25.0, 5.0, step=0.5)
demand = st.sidebar.slider("Local Demand Score (1-10)", 1, 10, 5)
supply = st.sidebar.slider("Available Auto Drivers (1-10)", 1, 10, 5)

# Binary external variables
rain_input = st.sidebar.selectbox("Weather Condition", ["Clear Sky", "Heavy Rain Storm"])
fest_input = st.sidebar.selectbox("Day Classification", ["Normal Day", "Festival Day"])
extra_input = st.sidebar.selectbox("Driver Tariff Dispute status", ["No Dispute", "Driver Asks Extra Cash"])

# Convert human inputs to raw operational binaries
is_raining = 1.0 if rain_input == "Heavy Rain Storm" else 0.0
is_festival = 1.0 if fest_input == "Festival Day" else 0.0
asks_extra = 1.0 if extra_input == "Driver Asks Extra Cash" else 0.0

current_hour = datetime.now().hour
current_day = datetime.now().strftime('%A')

# 4. MAIN DASHBOARD INTERFACE LAYOUT
st.title("🛺 Coimbatore RideFlow AI Operations Room")
st.markdown("### **End-to-End Intelligent Ride Optimization System**")
st.divider()

# Only run analytics engine if background models loaded perfectly without errors
if "🟢" in model_status:
    # Create layout grid for metrics panels (4 Columns for our 4 core metrics)
    col1, col2, col3, col4 = st.columns(4)

    # 5. PRE-PROCESSING FEATURE PIPELINE (One-Hot Encoding Alignment)
    all_columns = [
        'Hour_of_Day', 'Distance_km', 'Demand_Score', 'Supply_Score', 'Is_Raining', 'Is_Festival',
        'ETA_minutes', 'Driver_Asks_Extra', 'Pickup_Zone_Peelamedu', 'Pickup_Zone_RS Puram',
        'Pickup_Zone_Saravanampatti', 'Pickup_Zone_Ukkadam', 'Drop_Zone_Peelamedu', 'Drop_Zone_RS Puram',
        'Drop_Zone_Saravanampatti', 'Drop_Zone_Ukkadam', 'Day_of_Week_Monday', 'Day_of_Week_Saturday',
        'Day_of_Week_Sunday', 'Day_of_Week_Thursday', 'Day_of_Week_Tuesday', 'Day_of_Week_Wednesday'
    ]

    # Construct base numeric dictionary
    input_data = {col: 0.0 for col in all_columns}
    input_data['Hour_of_Day'] = float(current_hour)
    input_data['Distance_km'] = float(distance)
    input_data['Demand_Score'] = float(demand)
    input_data['Supply_Score'] = float(supply)
    input_data['Is_Raining'] = is_raining
    input_data['Is_Festival'] = is_festival
    input_data['Driver_Asks_Extra'] = asks_extra

    # Set matching One-Hot encoded columns to True (1.0)
    if f"Pickup_Zone_{pickup}" in input_data: input_data[f"Pickup_Zone_{pickup}"] = 1.0
    if f"Drop_Zone_{drop}" in input_data: input_data[f"Drop_Zone_{drop}"] = 1.0
    if f"Day_of_Week_{current_day}" in input_data: input_data[f"Day_of_Week_{current_day}"] = 1.0

    # Convert query row to DataFrame format
    X_query = pd.DataFrame([input_data])

    # 6. RUN DEEP LEARNING MODEL STEP: TASK 1 (LSTM ETA CLOCK PREDICTION)
    dl_features = pd.DataFrame([[distance, demand, supply, is_raining, current_hour]])
    dl_features_scaled = scaler.transform(dl_features)
    X_lstm_query = np.reshape(dl_features_scaled, (1, 1, 5))

    predicted_eta = float(lstm_model.predict(X_lstm_query, verbose=0).flatten()[0])
    predicted_eta = max(1.0, round(predicted_eta, 1)) # Safety clip bounds

    # Inject computed real-time ETA feature back into ML pipeline input vector
    X_query['ETA_minutes'] = predicted_eta

    # 7. RUN MACHINE LEARNING MODEL STEPS: MODULE 1 (SURGE & CANCELLATION)
    predicted_surge = float(lr_model.predict(X_query))
    predicted_surge = max(1.0, min(2.2, round(predicted_surge, 2))) # Strict business boundaries

    # Base auto fare logic matching Coimbatore input specifications
    if distance <= 2:
        base_fare = 40.0
    elif distance <= 10:
        base_fare = 40.0 + (distance - 2) * 15.0
    else:
        base_fare = 40.0 + (8 * 15.0) + (distance - 10) * 14.0

    if (current_hour >= 22) or (current_hour < 6): base_fare *= 1.5
    if is_raining == 1.0: base_fare += 50.0

    final_computed_fare = round(base_fare * predicted_surge, 2)

    # Classification Step: Predict Cancellation Risk Probability
    cancellation_probability = float(rf_clf.predict_proba(X_query)[:, 1])

    # 8. INJECT INTERACTIVE METRIC VALUES ON STREAMLIT DASHBOARD
    with col1:
        st.metric(label="⏱️ LSTM Predicted ETA", value=f"{predicted_eta} Mins")
    with col2:
        st.metric(label="📈 Dynamic Surge Multiplier", value=f"{predicted_surge}x")
    with col3:
        st.metric(label="💰 Final Computed Fare", value=f"₹{final_computed_fare}")
    with col4:
        st.metric(label="🚨 Cancellation Risk Probability", value=f"{round(cancellation_probability * 100, 1)}%")

    st.divider()

    # 9. SYSTEM DIAGNOSTIC ANALYTICS OUTPUT CARDS
    st.subheader("🤖 Fleet Predictive Analytics Dispatch Summary")
    st.markdown(f"**Route Vector:** From `{pickup}` to `{drop}`")
    st.markdown(f"**Temporal Stamp:** Checked on `{current_day}` at Hour `{current_hour}:00`")

    if cancellation_probability > 0.60:
        st.error("🚨 HIGH DISPATCH RISK: System flags an elevated risk of driver cancellation due to current tariff disputes or long route parameters.")
    else:
        st.success("✨ OPTIMAL ROUTE: Low cancellation probability. Fleet matching system confirms high success rate.")

else:
    st.error("Please ensure all pre-trained model files (.pkl, .h5) are present in the directory path.")
