# Welcome to the SuperDataScience Community Project!
Welcome to the **MLPayGrade: Predicting Salaries in the Machine Learning Job Market** repository! 🎉

This project is a collaborative initiative brought to you by SuperDataScience, a thriving community dedicated to advancing the fields of data science, machine learning, and AI. We are excited to have you join us in this journey of learning, experimentation, and growth.

To contribute to this project, please follow the guidelines avilable in our [CONTRIBUTING.md](CONTRIBUTING.md) file.

# Project Scope of Works:

## Project Overview
**MLPayGrade** is an end-to-end project built on 2024 salary data for machine learning-related roles. Participants will explore global salary trends, analyze the effect of job features on pay, and build models to predict salaries.

This project is split into two tracks to cater to different skill levels:
- 🟢 **Beginner Track** – Traditional ML pipeline with Streamlit deployment
- 🔴 **Advanced Track** – Deep learning on tabular data with embeddings and model explainability

Link to dataset: https://www.kaggle.com/datasets/chopper53/machine-learning-engineer-salary-in-2024

## 🟢 Beginner Track: Machine Learning Salary Predictor

### Objectives
#### Exploratory Data Analysis
- Clean and explore the dataset
- Encode categorical variables and handle outliers
- Normalize salary values for better model stability

**Key Questions to Answer**:
- What roles or experience levels yield the highest average salary?
- Does remote work correlate with higher or lower salaries?
- Are there differences in salary based on company size or location?
- How consistent are salaries across similar job titles?

#### Model Development
- Train multiple regression models: Linear Regression, Random Forest, and XGBoost
- Use `salary_in_usd` as the target and features like job title, location, remote ratio, and experience
- Track all model experiments and hyperparameters using **MLflow**
- Evaluate using RMSE, MAE, and R²
- Select and fine-tune the best model

#### Model Deployment
- Build a Streamlit app to allow users to input job attributes and receive a salary prediction
- Deploy the model to Streamlit Community Cloud

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `mlflow`
- **Deployment**: `streamlit`
---

## 🔴 Advanced Track: Deep Learning for Tabular Compensation Prediction

### Objectives
#### Exploratory Data Analysis
- Analyze salary distributions across categorical groupings
- Assess skewness, feature sparsity, and potential interaction terms
- Engineer new statistical features

**Key Questions to Answer**
- Which features most strongly influence salary distribution?
- Do certain job titles or experience levels benefit more from remote work?
- What is the variance in salaries within the same employment type and company size?
- Can we identify outlier salary records, and should they be retained or removed?

#### Model Development
- Design and train a **feedforward neural network (FFNN)** using **PyTorch** or **TensorFlow**
- Use **embedding layers** for high-cardinality categoricals (job title, location, etc.)
- Apply dropout, batch normalization, ReLU activations, and early stopping
- Track all model configurations, metrics, and hyperparameters using **MLflow**
- Evaluate using MAE, RMSE, R²; include residual analysis
- Optional: compare with LightGBM or CatBoost baseline

#### Explainability

- Use **SHAP** or **Permutation Importance** to understand feature contributions
- Visualize SHAP summaries and highlight how feature changes affect predictions

#### Model Deployment
- Deploy the model in a Streamlit app
- Accept inputs for job configuration, return prediction and SHAP-based interpretation
- Host the model on Streamlit Community Cloud or Hugging Face Spaces (if needed)

### Technical Requirements
- **Data Handling & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Deep Learning**: `tensorflow` or `pytorch`, `mlflow`
- **Explainability**: `shap`, `scikit-learn`
- **Deployment**: `streamlit`

---

## Workflow & Timeline (Both Tracks)

| Phase                     | Core Tasks                                                                           | Duration      |
| ------------------------- | ------------------------------------------------------------------------------------ | ------------- |
| **1 · Setup + EDA**       | Set up repo and environment; explore and clean the dataset; answer key EDA questions | **Week 1**    |
| **2 · Model Development** | Build and tune ML or DL models, evaluate performance, track experiments with MLflow  | **Weeks 2–4** |
| **3 · Deployment**        | Build Streamlit app and deploy model to cloud                                        | **Week 5**    |


