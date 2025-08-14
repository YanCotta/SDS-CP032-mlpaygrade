import streamlit as st
import pandas as pd
import joblib
import logging
from pathlib import Path

MODELDIR = Path("submissions/team-members/yan-cotta/models")

@st.cache_resource(show_spinner=False)
def load_models():
    pipe = joblib.load(MODELDIR / "xgb_pipeline.pkl")
    mapie = None
    try:
        mapie = joblib.load(MODELDIR / "xgb_mapie.pkl")
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.error(f"Error loading xgb_mapie.pkl: {e}")
    return pipe, mapie

st.set_page_config(
    page_title="MLPayGrade", 
    page_icon="💼",
    layout="wide"
)

st.title("💼 MLPayGrade")
st.markdown("**Professional salary prediction with realistic uncertainty estimates**")

pipe, mapie = load_models()

# Sidebar for inputs
st.sidebar.header("Enter Job Details")

job_category = st.sidebar.selectbox(
    "Job Category", 
    ["DATA_SCIENCE","DATA_ENGINEERING","DATA_ANALYSIS","MACHINE_LEARNING","MANAGEMENT","SPECIALIZED"],
    help="Primary job function category"
)

continent = st.sidebar.selectbox(
    "Continent", 
    ["NORTH_AMERICA","EUROPE","ASIA_PACIFIC","OTHER"],
    help="Geographic region of the company"
)

experience_level = st.sidebar.selectbox(
    "Experience Level", 
    ["EN","MI","SE","EX"],
    help="EN=Entry, MI=Mid, SE=Senior, EX=Executive"
)

employment_type = st.sidebar.selectbox(
    "Employment Type", 
    ["FT","PT","CT","FL"],
    help="FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance"
)

company_size = st.sidebar.selectbox(
    "Company Size", 
    ["S","M","L"],
    help="S=Small, M=Medium, L=Large"
)

work_year = st.sidebar.selectbox(
    "Work Year", 
    [2020,2021,2022,2023,2024],
    index=4,  # Default to 2024
    help="Year of employment"
)

remote_ratio = st.sidebar.slider(
    "Remote Work Ratio (%)", 
    0, 100, 100, step=50,
    help="Percentage of work done remotely"
)

X = pd.DataFrame([{
    "job_category": job_category,
    "continent": continent,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "company_size": company_size,
    "work_year": work_year,
    "remote_ratio": remote_ratio,
}])

if st.sidebar.button("Predict Salary", type="primary"):
    y_hat = float(pipe.predict(X)[0])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Annual Salary (USD)", 
            value=f"${y_hat:,.0f}"
        )

    with col2:
        if mapie:
            y_pred, y_int = mapie.predict(X, alpha=0.1)
            lo = float(y_int[0, 0])
            hi = float(y_int[0, 1])
            st.metric(
                label="90% Confidence Interval", 
                value=f"${lo:,.0f} - ${hi:,.0f}"
            )
            st.caption("Based on our model, we are 90% confident the actual salary falls between these bounds. The average width of this range is about $161K, reflecting the high natural salary variability in the market.")
        else:
            st.info("Confidence intervals unavailable")

# About section
st.markdown("---")
st.subheader("About This Model")

st.markdown("""
**Model Details:**
- **Algorithm**: XGBoost pipeline trained on 2020-2024 salary data
- **Realistic Performance**: Test MAE of ~$48.5K (2024 data)
- **Training Approach**: Leak-proof pipeline with temporal splits to prevent data leakage
- **Data Coverage**: 16,494 salary records from the ML/Data Science job market

**Uncertainty & Limitations:**
- This model has a realistic Mean Absolute Error of ~$48.5K, reflecting the inherent salary variability in the market
- Wide confidence intervals indicate high natural variation in compensation for similar profiles
- Predictions should be used as rough guidance rather than precise estimates
- Model performance aligns with the intrinsic "ceiling" of salary predictability (~$42.7K baseline)

**Reliability Note:** This project prioritizes honest uncertainty quantification over artificially low error metrics.
""")

st.markdown("---")
st.caption("MLPayGrade • Advanced Track • Final Production Model")
