# MLPayGrade Advanced Track – Chiti Nkhuwa

## Project Overview
This directory contains my work for the Advanced Track of the MLPayGrade project, focused on predicting salaries in the machine learning job market using 2024 data.

## Current Progress
### 1. Exploratory Data Analysis (EDA)
- **Data Cleaning:** Removed 6,401 duplicate rows, leaving 10,093 unique records. No missing values in key columns.
- **Salary Distribution:**
  - Right-skewed (skewness: 1.64, kurtosis: 8.39), with a long tail of high earners.
  - Outliers detected: 173 (1.7% of data) using IQR method.
  - ![Salary Distribution Histogram](images/salary_histogram.png) *(see notebook for plot)*
- **Experience Level:**
  - Executives (EX) and Seniors (SE) earn the most; clear salary progression by experience.
  - Average salaries: EX: $195,322, SE: $163,361, MI: $123,944, EN: $91,378.
  - ![Boxplot by Experience Level](images/boxplot_experience.png) *(see notebook for plot)*
- **Job Title:**
  - Top 10 most frequent job titles analyzed; Research Scientist, ML Engineer, and Data Architect are among the highest paid.
  - 155 unique job titles (high cardinality).
  - ![Boxplot by Job Title](images/boxplot_jobtitle.png) *(see notebook for plot)*
- **Remote Work:**
  - On-site and Remote roles have higher average salaries than Hybrid.
  - Average salaries: On-site: $150,968, Remote: $143,249, Hybrid: $83,087.
  - ![Boxplot by Remote Work Type](images/boxplot_remote.png) *(see notebook for plot)*
- **Company Size & Location:**
  - Medium companies pay the most on average; US, CA, AU lead in location-based salary.
  - 77 unique company locations (high cardinality).
  - ![Boxplot by Company Size](images/boxplot_companysize.png) *(see notebook for plot)*
- **Employment Type & Variance:**
  - Provided mean, median, std, and count for each employment type × company size group.
  - Full-time (FT) salaries: L: $122,976, M: $149,811, S: $87,956 (means).

## Next Steps
- Feature engineering (e.g., region buckets, seniority buckets, interaction terms)
- Encoding categorical variables for deep learning
- Outlier handling strategy for modeling
- Model development (FFNN with embeddings)
- Experiment tracking with MLflow
- Explainability (SHAP, permutation importance)
- Deployment (Streamlit app)

## Files
- `MLPaygrade-EDA.ipynb`: Main notebook for EDA and subsequent modeling steps.
- `images/`: Directory for saving EDA plots (create and save plots as PNGs for reference).

---

_This README will be updated as the project progresses to reflect new findings, modeling steps, and deployment details._ 