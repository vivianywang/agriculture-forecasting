import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import API_BASE_URL

RISK_COLORS = {"Low": "green", "Moderate": "orange", "High": "red"}

st.set_page_config(page_title="AI Wheat Forecaster for Canada", page_icon="🌾")
st.title("🌾 AI Wheat Forecaster for Canada")
st.caption("Predicts wheat yield, weather-driven risk, and a recommendation for any Canadian city.")

with st.form("predict_form"):
    city = st.text_input("City", placeholder="e.g. Regina")
    use_custom_year = st.checkbox("Predict for a specific year")
    year = st.number_input("Year", min_value=1995, max_value=2100, value=2025, step=1) if use_custom_year else None
    submitted = st.form_submit_button("Predict")

if submitted:
    if not city.strip():
        st.error("Enter a city name.")
    else:
        payload = {"city": city}
        if year is not None:
            payload["year"] = int(year)

        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the API at {API_BASE_URL}. Is it running? (uvicorn src.api.main:app --reload)")
            resp = None

        if resp is not None:
            if resp.status_code != 200:
                st.error(resp.json().get("detail", "Prediction failed."))
            else:
                result = resp.json()

                st.subheader(f"Field: {result['city']}, {result['province']}")

                col1, col2 = st.columns(2)
                col1.metric("Predicted Yield", f"{result['predicted_yield_t_ha']} t/ha")
                col2.metric("Predicted Yield", f"{result['predicted_yield_bu_ac']} bu/ac")

                st.markdown(f"**Prediction year:** {result['prediction_year']}")

                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                for col, label, key in [
                    (risk_col1, "Drought Risk", "drought_risk"),
                    (risk_col2, "Frost Risk", "frost_risk"),
                    (risk_col3, "Heat Stress Risk", "heat_stress_risk"),
                    (risk_col4, "Overall Risk", "overall_risk"),
                ]:
                    color = RISK_COLORS.get(result[key], "gray")
                    col.markdown(f"**{label}**")
                    col.markdown(f":{color}[{result[key]}]")

                st.progress(result["crop_health_score"] / 100, text=f"Crop Health Score: {result['crop_health_score']}/100")

                st.markdown("**Recommendation**")
                st.write(result["recommendation"])

                st.caption(result["weather_basis_note"])