import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


file=input('filename')

df=pd.read_csv(file)
print('Shape of the original data {}'.format(df.shape))

df.drop_duplicates(keep='first', inplace=True)
df = df.loc[df.work_year > 2021]
df.drop(['work_year','salary','salary_currency'], axis=1, inplace=True)
print('The shape of the data after removing old data and the unnecessary columns: {}'.format(df.shape))

df['remote_ratio'] = df.remote_ratio.astype('object')

#company_location
filter = {'AU': 0,
 'CA': 0,
 'DE': 3,
 'ES': 1,
 'FR': 3,
 'GB': 3,
 'IN': 1,
 'NL': 1,
 'US': 4,
 'other': 2}
df['company_location'] = df.company_location.apply(lambda x: 'other' if x not in filter.keys()  else x)
df.company_location = df.company_location.map(filter)

print('Company location fixed')

#job_title
filter = {'AI Developer': 0,
 'AI Scientist': 0,
 'Applied Data Scientist': 0,
 'Business Intelligence Engineer': 0,
 'Data Analytics Manager': 0,
 'Data Product Owner': 0,
 'Data Quality Engineer': 0,
 'MLOps Engineer': 0,
 'NLP Engineer': 0,
 'other': 0,
 'Analytics Engineer': 1,
 'Machine Learning Engineer': 1,
 'Machine Learning Scientist': 1,
 'Prompt Engineer': 1,
 'Research Engineer': 1,
 'Research Scientist': 1,
 'Business Intelligence': 2,
 'Business Intelligence Manager': 2,
 'Business Intelligence Specialist': 2,
 'Cloud Database Engineer': 2,
 'Data Integration Engineer': 2,
 'Data Management Specialist': 2,
 'Data Modeler': 2,
 'Data Operations Engineer': 2,
 'Data Product Manager': 2,
 'Data Science Engineer': 2,
 'Data Science Lead': 2,
 'Data Science Practitioner': 2,
 'Data Strategist': 2,
 'ETL Developer': 2,
 'Machine Learning Infrastructure Engineer': 2,
 'Machine Learning Operations Engineer': 2,
 'Machine Learning Researcher': 2,
 'Robotics Engineer': 2,
 'AI Engineer': 3,
 'Applied Scientist': 3,
 'Computer Vision Engineer': 3,
 'Data Analytics Lead': 3,
 'Data Architect': 3,
 'Data Lead': 3,
 'Data Science': 3,
 'Data Science Manager': 3,
 'Deep Learning Engineer': 3,
 'Director of Data Science': 3,
 'Head of Data': 3,
 'Head of Data Science': 3,
 'ML Engineer': 3,
 'Machine Learning Software Engineer': 3,
 'AI Programmer': 4,
 'BI Data Analyst': 4,
 'Data Analytics Consultant': 4,
 'Data Analytics Specialist': 4,
 'Data Developer': 4,
 'Data Management Analyst': 4,
 'Data Operations Analyst': 4,
 'Data Operations Associate': 4,
 'Data Quality Analyst': 4,
 'Encounter Data Management Professional': 4,
 'Insight Analyst': 4,
 'Lead Machine Learning Engineer': 4}

df['job_title'] = df.job_title.apply(lambda x: 'other' if x not in filter.keys() else x)
df.job_title = df.job_title.map(filter)
print('Job titles fixed')

# employee_residence
filter = {'DE': 0,
 'FR': 0,
 'GB': 0,
 'PL': 0,
 'UA': 0,
 'other': 0,
 'CA': 1,
 'US': 1,
 'AR': 2,
 'BR': 2,
 'ES': 2,
 'IN': 2,
 'IT': 2,
 'LT': 2,
 'LV': 2,
 'PT': 2,
 'ZA': 2,
 'AU': 3,
 'EG': 3,
 'MX': 4,
 'IE': 5,
 'CO': 6,
 'NL': 6}

df['employee_residence'] = df.employee_residence.apply(lambda x: 'other' if x not in filter.keys() else x)
df.employee_residence = df.employee_residence.map(filter)
print('Employee residence fixed')

#salary_in_usd
df = df[df.salary_in_usd > 0]  # Filter out rows with non-positive salary_in_usd
df.salary_in_usd = np.log(df.salary_in_usd)


df.reset_index(drop=True, inplace=True)

file2=input('filename')
df.to_csv(file2, index=False)
print('Data saved as {}'.format(file2))

