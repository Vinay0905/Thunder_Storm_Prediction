import streamlit as st
import requests
import json

# 1. Page Configuration (The "Premium" Look)
st.set_page_config(
    page_title="ThunderForecast Pro",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ ThunderForecast AI: Predictive Analytics")
st.markdown("---")

# 2. Main Layout - Split into Inputs and Results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Atmospheric Parameters")
    
    # We use columns inside columns to organize the 10 inputs
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        sweat = st.number_input("SWEAT Index", value=183.0)
        showalter = st.number_input("Showalter Index", value=6.6)
        lifted = st.number_input("Lifted Index", value=-1.6)
        k_index = st.number_input("K Index", value=27.0)
        totals = st.number_input("Totals Totals Index", value=41.0)
        
    with sub_col2:
        cape = st.number_input("CAPE (J/kg)", value=129.0)
        cine = st.number_input("CINE (J/kg)", value=-12.0)
        pw = st.number_input("Precipitable Water (mm)", value=38.0)
        thickness = st.number_input("1000-500 Thickness (m)", value=5756.0)
        plcl = st.number_input("PLCL (hPa)", value=962.0)

# 3. The Function to Talk to FastAPI
def get_prediction():
    # This matches the PredictionRequest schema we built earlier
    payload = {
        "sweat_index": sweat,
        "showalter_index": showalter,
        "lifted_index": lifted,
        "k_index": k_index,
        "totals_totals_index": totals,
        "cape": cape,
        "cine": cine,
        "precipitable_water": pw,
        "thickness_1000_500": thickness,
        "plcl": plcl
    }
    
    try:
        # We send a POST request to our local FastAPI server
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# 4. Results Section
with col2:
    st.subheader("Forecast Result")
    predict_btn = st.button("Generate Prediction", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Analyzing atmospheric data..."):
            result = get_prediction()
            
            if "error" in result:
                st.error(f"Error connecting to API: {result['error']}")
            else:
                prob = result['probability']
                pred = result['prediction']
                
                # Visual Indicators
                if pred == 1:
                    st.error("### 🌩️ THUNDERSTORM LIKELY")
                    st.warning(f"Probability: {prob*100:.1f}%")
                else:
                    st.success("### ☀️ NO THUNDERSTORM")
                    st.info(f"Probability: {prob*100:.1f}%")
                
                # Progress bar for probability
                st.progress(prob)
                
                # Extra Meteorological insight
                if prob > 0.4 and pred == 0:
                    st.write("📌 *Model Note: High instability detected, but below threshold.*")

st.markdown("---")
st.caption("Powered by XGBoost & FastAPI | Feature Engineering: Guardrail Enabled")
