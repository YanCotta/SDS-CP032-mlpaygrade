import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from lib.utils import job_type_label, leadership_label
from lib.utils import heatmap_by_category, alpha2_to_alpha3, features_preprocessing, log_to_mlflow

cwd = Path.cwd()
my_folder = Path(cwd)/"submissions/team-members/hien-nguyen"
source_data = my_folder/"data"
output = my_folder/"output"
model = joblib.load(output/'models/bagging_regressor.joblib')
df = pd.read_csv(source_data/"salaries.csv")
ref_df = pd.read_csv(output/'preprocessed_data/kmeans_gdp_reference.csv')
country_to_continent = {
    'ES': 'Europe', 'SI': 'Europe', 'IQ': 'Asia', 'IL': 'Asia', 'HK': 'Asia',
    'NG': 'Africa', 'LV': 'Europe', 'AS': 'North America', 'BO': 'South America',
    'AE': 'Asia', 'GH': 'Africa', 'QA': 'Asia', 'AU': 'Oceania', 'SE': 'Europe',
    'RS': 'Europe', 'JP': 'Asia', 'DK': 'Europe', 'IT': 'Europe', 'PR': 'North America',
    'AM': 'Asia', 'TH': 'Asia', 'NL': 'Europe', 'PE': 'South America', 'BG': 'Europe',
    'HN': 'North America', 'NO': 'Europe', 'MX': 'North America', 'KW': 'Asia',
    'VN': 'Asia', 'SA': 'Asia', 'GI': 'Europe', 'PK': 'Asia', 'BR': 'South America',
    'AD': 'Europe', 'BA': 'Europe', 'UG': 'Africa', 'LB': 'Asia', 'LU': 'Europe',
    'GR': 'Europe', 'BS': 'North America', 'EC': 'South America', 'IN': 'Asia',
    'CN': 'Asia', 'CL': 'South America', 'DZ': 'Africa', 'RU': 'Europe', 'HU': 'Europe',
    'FR': 'Europe', 'UZ': 'Asia', 'EE': 'Europe', 'IR': 'Asia', 'AT': 'Europe',
    'ID': 'Asia', 'AR': 'South America', 'SG': 'Asia', 'DO': 'North America',
    'GB': 'Europe', 'CR': 'North America', 'UA': 'Europe', 'PH': 'Asia', 'BE': 'Europe',
    'LT': 'Europe', 'NZ': 'Oceania', 'EG': 'Africa', 'CA': 'North America', 'IE': 'Europe',
    'OM': 'Asia', 'MY': 'Asia', 'KR': 'Asia', 'KE': 'Africa', 'TR': 'Asia',
    'GE': 'Asia', 'DE': 'Europe', 'CY': 'Europe', 'ZA': 'Africa', 'CH': 'Europe',
    'MT': 'Europe', 'MD': 'Europe', 'MU': 'Africa', 'US': 'North America',
    'HR': 'Europe', 'FI': 'Europe', 'CF': 'Africa', 'RO': 'Europe', 'CO': 'South America',
    'TN': 'Africa', 'JE': 'Europe', 'PL': 'Europe', 'CZ': 'Europe', 'PT': 'Europe'
}

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
Stop guessing—use data for accurate salary benchmarks.
""")
    st.markdown("""
    To estimate your predicted salary, just follow the steps below:
    1. Enter your job details on the left;
    2. Click "Predict" to see your estimated salary.
    """)

st.subheader("Your predicted salary:")

######################
#sidebar layout
######################

st.sidebar.title("Salary Prediction Input")
st.sidebar.image(output/'img/app_img2.png', width=100)
st.sidebar.write("Please enter the details that describe the job position and candidate")

# Input features based on your dataset
work_year = st.sidebar.selectbox("Select work year:", ("2025"))
experience_level = st.sidebar.selectbox("Choose experience level:",
    ("EN", "MI", "SE", "EX"))  # Entry, Mid, Senior, Executive
employment_type = st.sidebar.selectbox("Employment type:",
    ("FT", "PT", "CT", "FL"))  # Full-time, Part-time, Contract, Freelance
job_title = st.sidebar.selectbox("Job Title:", sorted(df['job_title'].unique()))
salary_currency = st.sidebar.selectbox("Salary currency:", sorted(df['salary_currency'].unique()))
employee_residence = st.sidebar.selectbox("Employee residence country:", sorted(df['employee_residence'].unique()))
remote_ratio = st.sidebar.slider("Remote ratio (percent):", 0, 100, 0, step=10)
company_location = st.sidebar.selectbox("Company location:", sorted(df['company_location'].unique()))
company_size = st.sidebar.selectbox("Company size:", ("S", "M", "L")) # Small/Medium/Large

def preprocess_salary(work_year, experience_level, employment_type, job_title, salary_currency,
                     employee_residence, remote_ratio, company_location, company_size):
    # Pre-processing user input
    user_input_dict = {
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'salary_currency': [salary_currency],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size]
    }
    user_input = pd.DataFrame(data=user_input_dict)

    user_input2 = user_input[['experience_level', 'company_size', 'work_year',
                             'remote_ratio', 'salary_currency', 'employment_type']]
    user_input2['job_type'] = user_input['job_title'].apply(job_type_label)
    user_input2['employee_continent'] = user_input['employee_residence'].map(country_to_continent)
    user_input2['company_continent'] = user_input['company_location'].map(country_to_continent)
    user_input2['job_level'] = user_input['job_title'].apply(leadership_label)
    user_input2['exp_level_job'] = user_input2['experience_level'] + '_' + user_input2['job_type']
    user_input2['exp_level_econtinent'] = user_input2['experience_level'] + '_' + user_input2['employee_continent']
    user_input2['job_level_exp'] = user_input2['job_level'] + '_' + user_input2['experience_level']
    user_input2['work_year_econtinent'] = user_input2['work_year'] + '_' + user_input2['employee_continent']
    user_input2['same_continent'] = (user_input['employee_continent'] == user_input['company_continent']).astype(int)
    job_title_freq = df['job_title'].value_counts().to_dict()
    user_input2['job_title_popularity'] = df['job_title'].map(job_title_freq)

    return user_input

user_input = preprocess_salary(work_year, experience_level, employment_type, job_title, salary_currency,
                              employee_residence, remote_ratio, company_location, company_size)

# Predict button
btn_predict = st.sidebar.button("Predict")
if btn_predict:
    pred = model.predict(user_input)
    st.subheader('Predicted Annual Salary:')
    st.success(f'${pred[0]:,.0f}')