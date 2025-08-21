import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lib.app_utils import preprocess_salary, country_to_continent, aggregate_features
import matplotlib.pyplot as plt
import seaborn as sns
import shap

cwd = Path.cwd()
my_folder = Path(cwd)/"submissions/team-members/hien-nguyen"
source_data = my_folder/"data"
output = my_folder/"output"
preprocessed_data = my_folder/"output/preprocessed_data"

ct = joblib.load(output/'models/column_transformer.joblib')
sc = joblib.load(output/'models/scaler.joblib')
model = joblib.load(output/'models/random_forest_regressor.joblib')
ref_df = pd.read_csv(output/'preprocessed_data/st_reference.csv')
salary_in_usd = np.load(preprocessed_data/'salary_in_usd.npy', allow_pickle=True)
job_titles = np.load(preprocessed_data/'job_titles.npy', allow_pickle=True)
salary_currency = np.load(preprocessed_data/'salary_currency.npy', allow_pickle=True)
employee_residence = np.load(preprocessed_data/'employee_residence.npy', allow_pickle=True)
company_location = np.load(preprocessed_data/'company_location.npy', allow_pickle=True)
feature_names = pd.read_csv(preprocessed_data/'final_features.csv', names=['feature','importance'], skiprows=1)
feature_names['aggregated_feature'] = feature_names['feature'].apply(aggregate_features)

############################################################################
#APP

st.set_page_config(
    page_title="Salary Prediction App",
    page_icon=output/'img/app_img1.jpg'
)

######################
#main page layout
######################

st.title("Salary Prediction")

col1, col2 = st.columns([1, 1])
with col1:
    st.image(output/'img/app_img1.jpg')
with col2:
    st.write("""
"Are you curious about your earning potential? 💰
This machine learning app estimates salary based on your experience and job details!
Find out if your salary offer matches your role, experience, and market trends.
Stop guessing, use data for accurate salary benchmarks.
""")
    st.markdown("""
    To estimate your predicted salary, just follow the steps below:
    1. Enter your job details on the left;
    2. Click "Predict" to see your estimated salary.
    """)

######################
#sidebar layout
######################

st.sidebar.title("Salary Prediction Input")
st.sidebar.image(output/'img/app_img2.png', width=100)
st.sidebar.write("Please enter the details that describe the job position and candidate.")
with st.sidebar.expander("Parameter Descriptions"):
    st.write("""
    **Experience Level:**
    - EN: Entry
    - MI: Mid
    - SE: Senior
    - EX: Executive

    **Employment Type:**
    - FT: Full-Time
    - PT: Part-Time
    - CT: Contract
    - FL: Freelance

    **Country Codes (ISO 3166-1 alpha-2):**
    - Andorra: AD
    - Argentina: AR
    - Armenia: AM
    - Australia: AU
    - Austria: AT
    - Bahamas: BS
    - Belgium: BE
    - Bosnia and Herzegovina: BA
    - Bolivia: BO
    - Brazil: BR
    - Bulgaria: BG
    - Canada: CA
    - Central African Republic: CF
    - Chile: CL
    - China: CN
    - Colombia: CO
    - Costa Rica: CR
    - Croatia: HR
    - Cyprus: CY
    - Czechia: CZ
    - Denmark: DK
    - Dominican Republic: DO
    - Egypt: EG
    - Ecuador: EC
    - Estonia: EE
    - Finland: FI
    - France: FR
    - Germany: DE
    - Ghana: GH
    - Gibraltar: GI
    - Greece: GR
    - Hong Kong: HK
    - Hungary: HU
    - India: IN
    - Indonesia: ID
    - Iran: IR
    - Iraq: IQ
    - Ireland: IE
    - Israel: IL
    - Italy: IT
    - Japan: JP
    - Jersey: JE
    - Kenya: KE
    - Kuwait: KW
    - Latvia: LV
    - Lebanon: LB
    - Lithuania: LT
    - Luxembourg: LU
    - Malaysia: MY
    - Malta: MT
    - Mexico: MX
    - Moldova: MD
    - Netherlands: NL
    - New Zealand: NZ
    - Nigeria: NG
    - Norway: NO
    - Oman: OM
    - Pakistan: PK
    - Peru: PE
    - Philippines: PH
    - Poland: PL
    - Portugal: PT
    - Puerto Rico: PR
    - Qatar: QA
    - Romania: RO
    - Russia: RU
    - Saudi Arabia: SA
    - Serbia: RS
    - Singapore: SG
    - Slovenia: SI
    - South Africa: ZA
    - South Korea: KR
    - Spain: ES
    - Sweden: SE
    - Switzerland: CH
    - Thailand: TH
    - Tunisia: TN
    - Turkey: TR
    - Uganda: UG
    - Ukraine: UA
    - United Arab Emirates: AE
    - United Kingdom: GB
    - United States: US
    - Uzbekistan: UZ
    - Vietnam: VN
    """)

# Input features based on your dataset
# work_year = st.sidebar.selectbox("Select work year:", ("2024"))
experience_level = st.sidebar.selectbox("Choose experience level:",
    ("EN", "MI", "SE", "EX"))  # Entry, Mid, Senior, Executive
employment_type = st.sidebar.selectbox("Employment type:",
    ("FT", "PT", "CT", "FL"))  # Full-time, Part-time, Contract, Freelance
job_title = st.sidebar.selectbox("Job Title:", sorted(job_titles))
salary_currency = st.sidebar.selectbox("Salary currency:", sorted(salary_currency))
employee_residence = st.sidebar.selectbox("Employee residence country:", sorted(employee_residence))
remote_ratio = st.sidebar.selectbox("Remote ratio (percent):", (0, 50, 100))
company_location = st.sidebar.selectbox("Company location:", sorted(company_location))
company_size = st.sidebar.selectbox("Company size:", ("S", "M", "L")) # Small/Medium/Large

user_input = preprocess_salary(experience_level, employment_type, job_title, salary_currency,
                              employee_residence, remote_ratio, company_location, company_size,
                              country_to_continent, ref_df, ct, sc)

# Predict button
btn_predict = st.sidebar.button("Predict")
st.sidebar.markdown(
    "[🔗 View source code & full report on GitHub](https://github.com/HienNguyen2311/SDS-CP032-mlpaygrade/tree/hien-nguyen/submissions/team-members/hien-nguyen)"
)

######################
#Prediction part
######################

if btn_predict:
    pred = model.predict(user_input)
    pred_usd = np.exp(pred)
    st.subheader('Predicted Annual Salary:')
    st.success(f'${pred_usd[0]:,.0f}')

    ######################
    #visualization
    ######################

    st.subheader("Your Predicted Salary vs. Global Market")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(salary_in_usd, bins=40, kde=True, color='skyblue', ax=ax)
    ax.axvline(pred_usd[0], color='crimson', linewidth=2, label='Your Prediction')
    ax.set_xlabel('Annual Salary (USD)', fontsize=12)
    ax.set_ylabel('Number of Entries', fontsize=12)
    ax.set_title('Comparison to Global Market', fontsize=14)
    ax.legend()
    sns.despine()
    st.pyplot(fig)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(user_input)
    shap_vector = shap_values.values[0]  # SHAP values for one prediction (numpy vector)
    shap_df = pd.DataFrame({
    'Feature': feature_names['aggregated_feature'],
    'SHAP Value': shap_vector
    })
    shap_df = shap_df.groupby(['Feature'])['SHAP Value'].sum().reset_index()
    shap_df_sorted = shap_df.reindex(shap_df['SHAP Value'].sort_values(ascending=False).index)
    plt.figure(figsize=(8,5))
    sns.barplot(y='Feature', x='SHAP Value', data=shap_df_sorted.iloc[:5], palette='coolwarm')
    plt.title("SHAP Feature Contributions")
    plt.tight_layout()
    st.pyplot(plt)

    with st.expander("ℹ️ Important Notes for Using This Salary Prediction App"):
        st.markdown("""
    **Prediction Scale**
    - The app’s model predicts **log-transformed annual salary**. This allows more stable modeling, especially for high-value outliers. However, it means that all model explanations (feature impact, error metrics) are relative to the logarithm of salary.
    - Your final result is always shown as a salary in USD (converted back from the log scale), so you don’t need to do any conversions.

    **SHAP Feature Explanations**
    - The SHAP bar chart shows the relative importance of your input choices (such as experience level, location, job title, currency, etc.) on the predicted salary.
    - Behind the scenes, your selections are processed and encoded (sometimes as groups of columns), and the app **aggregates SHAP values** so that you always see the influence at the level of your own input fields, not just technical model features.
    - Interaction features (like Experience Level + Location) may appear. These arise when the model finds that certain combinations matter more than individual fields.
    - SHAP values reflect influence in the log-salary space. For reference, a SHAP value of 0.1 equates to about a 10% effect on predicted salary.

    **Prediction Error and Model Accuracy**
    - Multiple models were evaluated on the data; accuracy metrics are reported in both log-salary and dollar (USD) terms:
        - *Log Scale (lower is better):*
            - RMSE ranges 0.33–0.36 (roughly 33%–36% error in log-\$ units)
            - R² values around 0.5 (meaning about half of the variance in log-salary is explained)
        - *Original USD Scale (lower is better):*
            - RMSE (Standard Error) is about \$51,700–\$55,000
            - MAE (Average Error) is around \$38,000–\$40,000
    - The performance was achieved by the Random Forest Regressor model.
    - **Predictions are estimates:** There can be substantial differences between predicted and actual salaries, especially for rare combinations of features or highly variable job titles.

    **Dataset Range and Use**
    - The model is trained on data from a wide variety of tech/data roles globally, with salaries ranging from \$15,000 to \$800,000, 25th percentile at \$101,516, 75th percentile at \$185,900 and a mean of \$149,714.
    - The more your inputs look like the training data, the more the model is “in its comfort zone,” yielding better and more reliable predictions. If your situation is less common, results may be less accurate, and you should interpret them as informative estimates rather than absolute truths.

    **Features Used in the Model**
    - The model automatically extracts and combines information from your input, including composite and interaction features that help represent common real-world patterns in pay.
    - Not all subtle, personal, or local factors (like negotiation, company-specific bonuses, cost-of-living quirks) are included.

    **Work Year Assumption**
    - The app currently assumes your work year is always 2024, because the underlying dataset and model only include information up to 2024, and there is no data available for 2025 or beyond.
    - If you are planning for a future year, please interpret your results as estimates based on the most up-to-date available data.


    **Interpretation and Cautions**
    - Use predictions and explanations as **guidance**, not as guaranteed salary offers.
    - The app is meant for benchmarking, educational, and orientation purposes, and does not guarantee employer offers.
    - Feature contributions and salary band estimates are most meaningful when combined with your own research and understanding of your market.
    - If a field is not listed in your input, it’s not influencing your particular prediction for this session.

    **Want to Know More?**
    - For complete details about the dataset, model logic, explanation, visualizations, and technical documentation, visit the associated GitHub repository.
    - [🔗 View source code & full report on GitHub](https://github.com/HienNguyen2311/SDS-CP032-mlpaygrade/tree/hien-nguyen/submissions/team-members/hien-nguyen)
        """)


