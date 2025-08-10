import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODELDIR = Path("submissions/team-members/yan-cotta/models")

@st.cache_resource(show_spinner=False)
def load_models():
    pipe = joblib.load(MODELDIR / "xgb_pipeline.pkl")
    mapie = None
    try:
        mapie = joblib.load(MODELDIR / "xgb_mapie.pkl")
    except Exception:
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.error(f"Error loading xgb_mapie.pkl: {e}")
    return pipe, mapie

st.set_page_config(page_title="MLPayGrade", page_icon="💼")
st.title("MLPayGrade — Salary Prediction with Uncertainty")

pipe, mapie = load_models()

cols = st.columns(3)
job_category = cols[0].selectbox("Job Category", ["DATA_SCIENCE","DATA_ENGINEERING","DATA_ANALYSIS","MACHINE_LEARNING","MANAGEMENT","SPECIALIZED"])
continent = cols[1].selectbox("Continent", ["NORTH_AMERICA","EUROPE","ASIA_PACIFIC","OTHER"])
experience_level = cols[2].selectbox("Experience Level", ["EN","MI","SE","EX"])

cols = st.columns(3)
employment_type = cols[0].selectbox("Employment Type", ["FT","PT","CT","FL"])
company_size = cols[1].selectbox("Company Size", ["S","M","L"])
work_year = cols[2].selectbox("Work Year", [2020,2021,2022,2023,2024])

remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 100, step=50)

X = pd.DataFrame([{
    "job_category": job_category,
    "continent": continent,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "company_size": company_size,
    "work_year": work_year,
    "remote_ratio": remote_ratio,
}])

if st.button("Predict"):
    y_hat = float(pipe.predict(X)[0])
    st.metric("Predicted salary (USD)", f"${y_hat:,.0f}")

    if mapie:
        y_pred, y_int = mapie.predict(X, alpha=0.1)
        lo = float(y_int[0, 0])
        hi = float(y_int[0, 1])
        st.write(f"90% interval: ${lo:,.0f} — ${hi:,.0f}")

    st.caption("Estimates reflect historical variability for similar profiles. Expect wide intervals due to intrinsic noise.")
