import sys
from pathlib import Path

import pandas as pd
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

                hist_resp = requests.get(f"{API_BASE_URL}/history", params={"city": city}, timeout=10)
                if hist_resp.status_code == 200:
                    hist = hist_resp.json()["history"]
                    hist_df = pd.DataFrame(hist)

                    predicted_row = pd.DataFrame([{
                        "year": result["prediction_year"],
                        "yield_t_ha": result["predicted_yield_t_ha"],
                        "mean_temp_c": None,
                        "total_precip_mm": None,
                    }])

                    st.markdown("### Historical Yield")
                    yield_chart_df = pd.concat([hist_df[["year", "yield_t_ha"]], predicted_row[["year", "yield_t_ha"]]])
                    yield_chart_df = yield_chart_df.set_index("year")
                    st.line_chart(yield_chart_df, y="yield_t_ha")
                    st.caption(f"Solid line is actual historical yield through {hist_df['year'].max()}; the final point is the {result['prediction_year']} prediction.")

                    st.markdown("### Historical Weather")
                    weather_col1, weather_col2 = st.columns(2)
                    with weather_col1:
                        st.line_chart(hist_df.set_index("year"), y="mean_temp_c")
                        st.caption("Mean growing-season temperature (°C)")
                    with weather_col2:
                        st.line_chart(hist_df.set_index("year"), y="total_precip_mm")
                        st.caption("Total growing-season precipitation (mm)")