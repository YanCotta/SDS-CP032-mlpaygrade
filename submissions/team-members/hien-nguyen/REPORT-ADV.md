# 📄 MLPayGrade – Project Report - 🔴 **Advanced Track**

Welcome to your personal project report!
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What roles or experience levels yield the highest average salary?

* New variable: job_type groups (Data, BI, ML, AI, Robotics, Others) is created from job_title.
* Figure 1 show that AI and ML roles tended to have higher median salaries, with significant outliers among "Others" and some "Data" jobs.
* EX (Executive) and SE (Senior) consistently earn most; EN (Entry) the least. Median salary line plot over years (Fifure 2) confirms EX>SE>MI>EN.

![Salaries Distribution by Category](img/all_features_boxplots.png)
Figure 1: Salaries Distribution by Category

![Median Salaries by Job Experience over the Years](img/median_salaries_lineplot.png)
Figure 2: Median Salaries by Job Experience over the Years

### 🔑 Question 2: Does remote work correlate with higher or lower salaries?

* Figure 1 shows 100% remote roles generally have higher median salaries, but with greater spread and more outliers.
* Heatmap of experience_level vs remote_ratio median salary (Figure 3) also confirms that, especially at Executive level, 100% remote fetches higher median salary.

![Median Salaries by Job Experience and Remote Work](img/median_salaries_heatmap.png)
Figure 3: Median Salaries by Job Experience and Remote Work

### 🔑 Question 3: Are there differences in salary based on company size or location?

* Company Size: M (median at 143K) > L (136K) > S (71K) in typical median salary (Figure 1).
* new variable: company_location is grouped by continent.
* Continent: North America offers highest salaries on average at 147K, then Oceania (115K) and Europe (69K), with Asia/Africa/South America much lower.
* Company Continent vs Salary: Consistently large differences, confirmed both by boxplot (Figure 1) and heatmap (Figure 3).

### 🔑 Question 4: How consistent are salaries across similar job titles?

* The bar chart of average salary by job title (Figure 4) shows a wide spread, with some job titles (such as specialized leadership or high-demand technical roles) earning much more than others.
* Some similar job titles have much higher average salaries but also higher variance, for example Data Scienctist (150K) and Data Science Consultant (112K).
* The boxplots of salaries by job type (Figure 1) underline an inconsistency: nearly every job type shows a large interquartile range, with numerous outliers both above and below the median. Job types such as Data, ML and AI feature a broad "box" and many dots, emphasizing a lack of consistency.

![Median Salary by Job Title](img/title_salary_barplot.png)
Figure 4: Median Salary by Job Title

---

## ✅ Week 2: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
**Which categorical features are high-cardinality, and how will you encode them for use with embedding layers?**  

* High-cardinality features: job_title, employee_residence, company_location
* These variables are dropped from initial modeling due to difficulties for tree models
* They are both grouped manually based on similar job title name and geography. Figure 5 presents how employee_residence is grouped based on geography.
* job_title and employee_residence are used as categorical inputs to find patterns in salary distribution. Instead of using the raw high-cardinality values directly in the model, I first encoded them with a OneHotEncoder and then applied KMeans clustering to group similar combinations of job title and location together into 10 groups. This new feature acts as a proxy for job–location salary patterns, enabling models to leverage grouped salary behavior without overfitting to individual job titles or locations. Figure 6 is a world map of instances grouped using KMeans.
* For other categoricals, I use a mix of OneHotEncoder (for nominal) and OrdinalEncoder (for ordered categories such as experience_level, company_size, work_year, remote_ratio).

![Countries Grouped by Continent](img/geogroup_map.png)
Figure 5: Countries Grouped by Continent

![Countries Grouped by KMeans Clusters](img/knngroup_map.png)
Figure 6: Countries Grouped by KMeans Clusters

---

#### 🔑 Question 2:
**What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?**  

* All continuous numerics are standardized via StandardScaler.
* For targets (salary_in_usd), I also use a log transform (log1p) to reduce skew and stabilize variance.
* log_salaries are much closer to normal, easing regression (Figure 7).

![Log Salary Distribution](img/logsalaries_histogplot.png)
Figure 7: Log Salary Distribution

---

#### 🔑 Question 3:
**Did you create any new features based on domain knowledge or data relationships? If yes, what are they and why might they help the model predict salary more accurately?**  

* job_level: Extracted seniority/leadership (via leadership_label), with 3 levels: Staff, Manager, Head
* Experience by Job Type: exp_level_job
* Employee/Company continent interaction: exp_level_econtinent, work_year_econtinent, same_continent
* Popularity: job_title_popularity (job title frequencies)
* Clustering: KMeans group of (employee_residence, job_title)
* Macro: Merged in GDP per country/year.
* Log Salary: Target transformed to log_salaries.

---

#### 🔑 Question 4:
**Which features, if any, did you choose to drop or simplify before modeling, and what was your justification?**  

* Dropped: job_title, employee_residence, and company_location (raw) — high cardinality, used their aggregates/clusters instead.
* Only categories with tractable cardinality were retained for encoding.
* Outliers not removed (explicitly quantified, not dropped) as dropping outliers did not improve performance metrics.

---

#### 🔑 Question 5:
**After preprocessing, what does your final input schema look like (i.e., how many numerical features and how many categorical features)? Are there any imbalance or sparsity issues to be aware of?**  


Final input:

* 4 ordinal features: experience_level, company_size, work_year, remote_ratio

* 12 nominal features: types, continents, engineered interactions, clusters.

* Target: log_salaries


Imbalance:

* Clear imbalance in all categorical features, by region, experience, job types, company size (see Figure 8).

* Some features (like groupings or clusters) may remain sparse, but overall the log target and aggregate encodings reduce extreme sparsity from high-cardinality categories.

![Categorical Value Counts by Features](img/all_features_barplots.png)
Figure 8: Categorical Value Counts by Features

---

### ✅ Week 3: Model Development & Experimentation

---

### 🔑 Question 1:
**What does your neural network architecture look like (input dimensions, hidden layers, activation functions, output layer), and why did you choose this structure?**  
🎯 *Purpose: Tests architectural understanding and reasoning behind model design.*

💡 **Hint:**  
Describe your FFNN: number of layers, units per layer, ReLU activations, batch normalization, dropout rates, etc.  
Explain why this architecture fits the tabular regression task (salary prediction).  
Include code and rationale for each component.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What loss function, optimizer, and evaluation metrics did you use during training? How did these choices influence your model’s learning behavior?**  
🎯 *Purpose: Tests knowledge of training dynamics and evaluation strategy.*

💡 **Hint:**  
Use MSE or MAE as loss (for regression), Adam or SGD as optimizer.  
Track RMSE, MAE, R².  
Reflect on how your metric trends evolved over epochs — was the model improving or plateauing?

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How did your model perform on the training and validation sets across epochs, and what signs of overfitting or underfitting did you observe?**  
🎯 *Purpose: Tests ability to diagnose model behavior using learning curves.*

💡 **Hint:**  
Plot loss and metrics over time.  
Overfitting → validation error increases while training error decreases.  
Underfitting → both remain high.  
Include screenshots or plots and interpretation.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**How did your deep learning model compare to a traditional baseline (e.g., Linear Regression or XGBoost), and what might explain the difference in performance?**  
🎯 *Purpose: Encourages comparative evaluation and model introspection.*

💡 **Hint:**  
Train a traditional model and compare RMSE, MAE, R².  
If the FFNN underperforms, discuss whether you need more tuning, more data, or regularization.  
If it outperforms, explain what complex patterns it likely captured.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**What did you log with MLflow (e.g., architecture parameters, metrics, model versions), and how did tracking help guide your experimentation?**  
🎯 *Purpose: Tests reproducibility and experiment management awareness.*

💡 **Hint:**  
Track model depth, units, dropout rate, learning rate, validation scores.  
Show how MLflow helped you compare runs, spot trends, or pick the best model.  
Include run screenshots or output tables.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:

**Which neural network architecture and hyperparameters did you experiment with, and how did you decide what to tune?**
🎯 *Purpose: Tests understanding of architectural design and tuning for tabular deep learning.*

💡 **Hint:**
Tuning ideas include: number of layers, neurons per layer, dropout rate, batch size, learning rate.
Justify choices based on previous results and known model behaviors.

✏️ *Your answer here...*

---

### 🔑 Question 2:

**What tuning strategy did you follow (manual, scheduler-based, Optuna, etc.) and how did it help refine your model?**
🎯 *Purpose: Evaluates ability to apply efficient search strategies for optimization.*

💡 **Hint:**
Manual tuning = more control but slower.
Optuna/RandomSearch = broader coverage.
Schedulers = dynamic adjustment during training.
Explain tradeoffs and justify your approach.

✏️ *Your answer here...*

---

### 🔑 Question 3:

**How did you monitor and evaluate the effect of tuning on model performance? Which metrics did you track and how did they change?**
🎯 *Purpose: Tests analytical thinking and metric interpretation skills.*

💡 **Hint:**
Track F1, MAE, RMSE, or R² across validation.
Use MLflow to visualize trends.
Mention learning curves, performance on different splits, and early stopping if used.

✏️ *Your answer here...*

---

### 🔑 Question 4:

**What did MLflow reveal about your tuning experiments, and how did it help in selecting your final model configuration?**
🎯 *Purpose: Encourages tool-based reflection on the model development process.*

💡 **Hint:**
Discuss MLflow runs, comparison plots, parameter filtering.
Explain how it saved time or revealed overlooked patterns.

✏️ *Your answer here...*

---

### 🔑 Question 5:

**What is your final model setup (architecture + hyperparameters), and why do you believe it’s the most generalizable?**
🎯 *Purpose: Tests ability to justify final model selection with evidence and insight.*

💡 **Hint:**
Include a summary of hyperparameters, loss function, optimizer.
Explain why this configuration balances performance and simplicity.
Support decision with validation performance and interpretability where possible.

✏️ *Your answer here...*

---


## ✅ Week 5: Model Deployment

> Document your approach to building and deploying the Streamlit app, including design decisions, deployment steps, and challenges.

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---
