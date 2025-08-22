
import pandas as pd
from lib.utils import job_type_label, leadership_label
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

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

def preprocess_salary(experience_level, employment_type, job_title, salary_currency,
                     employee_residence, remote_ratio, company_location, company_size,
                     country_to_continent, ref_df, ct, sc):
    # Pre-processing user input
    user_input_dict = {
        'work_year': ['2024'],
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
    user_input2 = user_input.copy()
    user_input2['job_type'] = user_input['job_title'].apply(job_type_label)
    user_input2['employee_continent'] = user_input['employee_residence'].map(country_to_continent)
    user_input2['company_continent'] = user_input['company_location'].map(country_to_continent)
    user_input2['job_level'] = user_input['job_title'].apply(leadership_label)
    user_input2['exp_level_job'] = user_input2['experience_level'] + '_' + user_input2['job_type']
    user_input2['exp_level_econtinent'] = user_input2['experience_level'] + '_' + user_input2['employee_continent']
    user_input2['job_level_exp'] = user_input2['job_level'] + '_' + user_input2['experience_level']
    user_input2['work_year_econtinent'] = user_input2['work_year'] + '_' + user_input2['employee_continent']
    user_input2['same_continent'] = (user_input2['employee_continent'] == user_input2['company_continent']).astype(int)
    user_input2 = pd.merge(user_input2, ref_df,
                           on=['employee_residence', 'job_title'],
                           how='left')
    ref_lookup = ref_df.drop_duplicates(subset=['employee_residence']).set_index('employee_residence')
    for col in ['kmeans_group', 'job_title_popularity', 'GDP_USD']:
        lookup_vals = user_input2['employee_residence'].map(ref_lookup[col])
        # Only fill NaNs:
        if col in user_input2.columns:
            user_input2[col] = user_input2[col].combine_first(lookup_vals)
        else:
            user_input2[col] = lookup_vals
    user_input2 = user_input2[['experience_level', 'company_size', 'work_year',
                             'remote_ratio', 'salary_currency', 'employment_type',
                             'job_type', 'employee_continent', 'company_continent',
                             'job_level', 'exp_level_job', 'exp_level_econtinent',
                             'job_level_exp', 'work_year_econtinent', 'kmeans_group', 'same_continent',
                             'job_title_popularity', 'GDP_USD']]
    user_input2[['remote_ratio', 'kmeans_group']] = user_input2[['remote_ratio', 'kmeans_group']].astype(str)
    X = user_input2.to_numpy()
    X = ct.transform(X)
    print("shape", X.shape)
    X = sc.transform(X)
    return X

def aggregate_features(feature_name):
    if 'company_continent' in feature_name:
        return 'Company Location'
    elif 'employment_type' in feature_name:
        return 'Emplotment Type'
    elif "exp_level_econtinent" in feature_name:
        return 'Experience Level - Employee Residence Interaction'
    elif 'exp_level_job' in feature_name:
        return 'Experience Level - Job Type Interaction'
    elif 'job_level_exp' in feature_name:
        return 'Experience Level - Job Level Interaction'
    elif 'job_level' in feature_name:
        return 'Job Title'
    elif 'job_type' in feature_name:
        return 'Job Title'
    elif 'kmeans_group' in feature_name:
        return 'Employee Residence - Job Title KMeans Cluster'
    elif 'salary_currency' in feature_name:
        return 'Salary Currency'
    elif 'work_year_econtinent' in feature_name:
        return 'Employee Residence - Work Year Interaction'
    elif 'company_size' in feature_name:
        return 'Company Size'
    elif 'experience_level' in feature_name:
        return 'Experience Level'
    elif 'remote_ratio' in feature_name:
        return 'Remote Ratio'
    elif 'work_year' in feature_name:
        return 'Work Year'
    elif 'GDP_USD' in feature_name:
        return 'Employee Residence Country GDP'
    else:
        return "Job Title Popularity"