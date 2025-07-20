# MLPayGrade: Advanced Deep Learning Track
## Predicting Salaries in the Machine Learning Job Market

**Team Member**: Yan Cotta  
**Track**: Advanced (Deep Learning with Embeddings & Explainability)  
**Project Timeline**: 5 weeks (July 2025)  
**Status**: Week 1 Complete ✅

---

## 📋 Project Overview

This repository implements the **Advanced Track** of the MLPayGrade community project, featuring:
- **Deep Learning on Tabular Data** using TensorFlow/Keras
- **Embedding Layers** for high-cardinality categorical features
- **Model Explainability** using SHAP analysis
- **MLflow Experiment Tracking** for reproducible ML workflows
- **Streamlit Deployment** for interactive salary predictions

### 🎯 Advanced Track Objectives
1. **Exploratory Data Analysis**: Analyze salary distributions, feature relationships, and data quality
2. **Deep Learning Model**: Design feedforward neural networks with embedding layers
3. **Model Explainability**: Implement SHAP-based feature importance analysis
4. **Production Deployment**: Build and deploy Streamlit application

---

## 📊 Dataset Overview

**Source**: [Kaggle ML Engineer Salary Dataset 2024](https://www.kaggle.com/datasets/chopper53/machine-learning-engineer-salary-in-2024)

### Key Statistics
- **Total Records**: 16,494 salary entries
- **Time Period**: 2020-2024 (88% from 2023-2024)
- **Features**: 11 columns (4 numerical, 7 categorical)
- **Target Variable**: `salary_in_usd` (range: $15K - $800K)
- **Data Quality**: ✅ Zero missing values, excellent quality

### Feature Breakdown
| Feature | Type | Unique Values | Description |
|---------|------|---------------|-------------|
| `work_year` | Numerical | 5 | Employment year (2020-2024) |
| `experience_level` | Categorical | 4 | EN, MI, SE, EX |
| `employment_type` | Categorical | 4 | FT, PT, CT, FL |
| `job_title` | Categorical | 155 | Specific role titles |
| `salary_in_usd` | **Target** | - | USD-converted salary |
| `employee_residence` | Categorical | 88 | Employee country |
| `remote_ratio` | Numerical | 3 | 0%, 50%, 100% remote |
| `company_location` | Categorical | 77 | Company country |
| `company_size` | Categorical | 3 | S, M, L |

---

## 🔬 Week 1: Comprehensive EDA Analysis

### 🎯 Key Research Questions Answered

#### **1. Feature Influence on Salary Distribution**
- **Experience Level Impact**: Clear progression EN ($92K) → MI ($126K) → SE ($164K) → EX ($195K)
- **Company Size Effect**: Medium companies (M) dominate dataset, show highest mean salaries
- **Employment Type**: Full-time dominates (99.5%), contract work shows high variance
- **Statistical Significance**: All categorical features highly significant (p < 0.001)

#### **2. Remote Work & Role Interaction**
- **Executive Level**: Benefits most from 100% remote work ($211K vs $182K on-site)
- **Entry Level**: Performs better on-site ($88K vs $50K remote)
- **Mid/Senior Levels**: Prefer on-site arrangements for optimal compensation
- **Pattern**: Remote benefit increases with seniority level

#### **3. Salary Variance Analysis**
- **Highest Variance**: Contract work at small companies ($121K std dev)
- **Lowest Variance**: Freelance at small companies ($13K std dev)
- **Most Stable**: Full-time at small companies
- **Risk Assessment**: Contract roles show highest compensation volatility

### 📈 Distribution Analysis

#### **Target Variable Characteristics**
- **Distribution**: Highly right-skewed (skewness: 1.49)
- **Central Tendency**: Mean $150K, Median $141K
- **Outliers**: 284 records (1.72%) above $312K threshold
- **Transformation**: Log transformation reduces skewness to -0.67 (81% improvement)

#### **Outlier Analysis Results**
- **Profile**: Primarily Senior (75%) and Mid-level (14%) professionals
- **Roles**: ML Engineers, Data Scientists, Research Scientists
- **Companies**: 95% work at medium-sized companies
- **Decision**: **Retain outliers** - represent legitimate high-value market segments

### 🌍 Geographic & Temporal Patterns

#### **Geographic Distribution**
- **US Dominance**: ~70% of all records from US companies
- **International Spread**: 77 countries with severe class imbalance
- **Premium Markets**: Switzerland, Luxembourg, Denmark show highest salaries
- **Data Risk**: Geographic bias toward US market conditions

#### **Temporal Trends** 
- **Data Volume**: 2023 (52%), 2024 (37%), earlier years (11%)
- **Salary Growth**: Explosive growth 2021-2023 (+34.5%, +14.4%), plateau 2024 (-2.0%)
- **Volatility**: Standard deviation peaked in 2020, stabilized 2022-2023
- **Model Risk**: Temporal bias may affect future predictions

---

## ⚠️ Critical Data Quality Insights

### **Identified Biases & Risks**

1. **Temporal Bias**: 88% of data from 2023-2024
   - Risk: Overfitting to current market conditions
   - Mitigation: Include temporal features, year-based validation

2. **Geographic Bias**: US-centric dataset
   - Risk: Poor generalization to global markets  
   - Mitigation: Separate validation for non-US predictions

3. **Class Imbalance**: Severe imbalances in multiple features
   - Employment Type: 99.5% full-time
   - Job Titles: 155 categories, many with <10 samples
   - Mitigation: Aggressive consolidation, class weighting

4. **Currency Dominance**: 97%+ salaries in USD
   - Benefit: Minimal conversion noise
   - Risk: May not capture true international salary patterns

---

## 🔧 Strategic Feature Engineering Plan

### **1. Job Title Consolidation (155 → 6 categories)**
```python
consolidation_strategy = {
    'DATA_SCIENCE': ['Data Scientist', 'Research Scientist', 'Applied Scientist'],
    'DATA_ENGINEERING': ['Data Engineer', 'Analytics Engineer', 'ML Engineer'],  
    'DATA_ANALYSIS': ['Data Analyst', 'Business Intelligence Analyst'],
    'MACHINE_LEARNING': ['Machine Learning Engineer', 'Machine Learning Scientist'],
    'MANAGEMENT': ['Data Manager', 'Head of Data', 'Director of Data Science'],
    'SPECIALIZED': [remaining_rare_titles]
}
```

### **2. Geographic Hierarchy (77 → 8 categories)**
```python
geographic_strategy = {
    'continent': ['NORTH_AMERICA', 'EUROPE', 'ASIA_PACIFIC', 'OTHER'],
    'economic_tier': ['HIGH_INCOME', 'DEVELOPED', 'EMERGING', 'OTHER']
}
```

### **3. Experience-Company Interaction Features**
```python
interaction_features = {
    'career_stage': f"{experience_level}_{company_size}",
    'seniority_score': encoded_experience * company_size_weight
}
```

### **4. Target Variable Transformation**
- **Primary**: Log transformation (`log1p(salary_in_usd)`)
- **Justification**: Reduces skewness by 81%, optimal for neural networks
- **Implementation**: Transform target, use `expm1()` for predictions

---

## 🚀 Week 2-5 Implementation Roadmap

### **Week 2: Feature Engineering & Data Preprocessing**
- [ ] Implement consolidation strategies
- [ ] Create interaction features  
- [ ] Handle class imbalances
- [ ] Set up train/validation/test splits with temporal considerations

### **Week 3-4: Deep Learning Model Development**
- [ ] Design feedforward neural network architecture
- [ ] Implement embedding layers for categorical features
- [ ] Set up MLflow experiment tracking
- [ ] Hyperparameter tuning with cross-validation
- [ ] Compare with baseline models (LightGBM, CatBoost)

### **Week 5: Explainability & Deployment**
- [ ] SHAP analysis for feature importance
- [ ] Model interpretation and business insights
- [ ] Streamlit application development
- [ ] Cloud deployment (Streamlit Community Cloud)

---

## 🏗️ Technical Stack

### **Core Libraries**
- **Data Science**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Deep Learning**: `tensorflow`, `keras`
- **Experiment Tracking**: `mlflow`
- **Explainability**: `shap`
- **Machine Learning**: `scikit-learn`
- **Deployment**: `streamlit`

### **Development Environment**
- **Python**: 3.12.3
- **Virtual Environment**: `.venv` (isolated dependencies)
- **Notebook**: Jupyter Lab with configured kernel
- **Version Control**: Git with `.gitignore` for data files

---

## 📁 Project Structure

```
yan-cotta/
├── mlpaygrade_advanced_track.ipynb    # Main analysis notebook
├── archive/
│   └── salaries.csv                   # Dataset (gitignored)
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
└── (future files)
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── model_architecture.py
    │   └── shap_analysis.py
    ├── models/                        # Trained model artifacts
    ├── experiments/                   # MLflow experiments
    └── streamlit_app.py              # Deployment application
```

---

## 📊 Model Performance Expectations

### **Success Metrics**
- **Primary**: RMSE on log-transformed target
- **Secondary**: MAE, R² score
- **Evaluation**: Residual analysis, prediction intervals
- **Baseline**: Linear regression, Random Forest, XGBoost

### **Validation Strategy**
- **Temporal Split**: Train on 2020-2023, validate on 2024
- **Geographic Split**: Separate validation for US vs international
- **Stratified Sampling**: Maintain rare category representation

---

## 🎯 Expected Business Impact

### **Use Cases**
1. **Job Seekers**: Salary expectations based on role, experience, location
2. **Employers**: Competitive salary benchmarking
3. **Career Planning**: Understanding factors that maximize compensation
4. **Market Analysis**: Trends in ML/Data Science compensation

### **Deliverables**
1. **Interactive Web App**: Real-time salary predictions with confidence intervals
2. **Explainable AI**: SHAP-based feature importance for transparent decisions
3. **Market Insights**: Comprehensive analysis of salary drivers in ML industry
4. **Reproducible Pipeline**: MLflow-tracked experiments for continuous improvement

---

## 🏆 Advanced Track Differentiation

Unlike the beginner track, this implementation features:
- **Deep Learning**: Neural networks with embedding layers vs traditional ML
- **Explainability**: SHAP analysis vs basic feature importance
- **Advanced EDA**: Statistical testing, temporal analysis, bias assessment
- **Production Ready**: MLflow tracking, proper validation, deployment considerations

---

**📧 Contact**: Yan Cotta at yanpcotta@gmail.com
**🔗 Repository**: [SDS-CP032-mlpaygrade](https://github.com/YanCotta/SDS-CP032-mlpaygrade)  
**📅 Last Updated**: July 9, 2025
