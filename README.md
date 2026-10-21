<div align="center">

📊 ChurnGuard AI

Customer Churn Prediction System

An End-to-End Machine Learning Project

Predicting customer churn through data analysis, model comparison, hyperparameter tuning, and an interactive prediction system.

Python
Scikit-learn
Pandas
FastAPI
Streamlit

</div>

────────

📑 Table of Contents

• 🎯 Overview
• ✨ Key Features
• 📊 Dataset
• 🔎 Exploratory Data Analysis
• ⚙️ Data Preparation
• 🤖 Models and Methodology
• 📈 Model Comparison
• 🏆 Final Model Evaluation
• 🏗️ System Architecture
• 🛠️ Tech Stack
• 📂 Project Structure
• 🚀 Setup and Usage
• ⚠️ Limitations and Future Work
• 👩‍💻 Author

────────

🎯 Overview

Customer churn happens when customers stop using a company’s services. Identifying customers who may churn can help businesses understand customer behavior and make more informed retention decisions.

ChurnGuard AI is a Machine Learning project that analyzes telecommunications customer data and predicts whether a customer is likely to churn. The workflow covers exploratory data analysis, data cleaning, feature engineering, model comparison, hyperparameter optimization, and evaluation. The project also includes a FastAPI backend and a Streamlit interface for interactive predictions.

✨ Key Features

• 📊 Exploratory Data Analysis (EDA) and customer behavior analysis
• 🧹 Duplicate investigation and data cleaning
• ⚙️ Feature engineering with Avg_Usage_Duration
• 🤖 Comparison of Logistic Regression, Random Forest, and Gradient Boosting
• 🔍 Stratified 5-fold cross-validation
• 🎛️ Random Forest hyperparameter tuning with RandomizedSearchCV
• 📈 Evaluation with Precision, Recall, F1-score, ROC-AUC, and PR-AUC
• 💾 Saving the selected model with Joblib
• ⚡ FastAPI backend and Streamlit interface for predictions

📊 Dataset

The dataset contains telecommunications customer information, usage behavior, and the target label Churn.

|Dataset Detail                                         |Value                           |
|-------------------------------------------------------|-------------------------------:|
|Original rows                                          |3,150                           |
|Original columns                                       |14                              |
|Missing values                                         |0                               |
|Exact duplicate rows initially identified              |300                             |
|Rows after duplicate/conflicting-label cleaning        |2,822                           |
|Features used before excluding `Status` for final model|14, including engineered feature|
|Target                                                 |`Churn`                         |

Target Distribution

|Churn label|Meaning|Original records|Percentage|
|----------:|-------|---------------:|---------:|
|`0`        |Stayed |2,655           |84.29%    |
|`1`        |Churned|495             |15.71%    |

The target classes are imbalanced, so accuracy is not used as the only measure of model quality.

🔎 Exploratory Data Analysis

The notebook explores customer behavior and its relationship with churn, including:

• Churn distribution
• Complaints and churn rate
• Subscription length
• Usage behavior: seconds of use, frequency of use, and SMS frequency
• Customer value
• Age group and tariff plan
• Customer status and churn
• Feature correlations
• Duplicate records and data-quality checks

Example Finding: Complaints

|Complaint status  |Customers|Churned customers|Churn rate|
|------------------|--------:|----------------:|---------:|
|No complaint (`0`)|2,909    |295              |10.14%    |
|Complaint (`1`)   |241      |200              |82.99%    |

In this dataset, customers with complaints have a substantially higher observed churn rate. This is an association in the data, not proof that complaints alone cause churn.

⚙️ Data Preparation

1. Data Quality Checks

• Checked missing values and found none in the original dataset.
• Investigated exact duplicate rows.
• Checked for identical feature combinations associated with conflicting churn labels.
• Removed exact duplicates and excluded conflicting feature groups from the cleaned modeling dataset.

2. Feature Engineering

Created Avg_Usage_Duration to represent average usage duration per usage frequency:

```python
Avg_Usage_Duration = Seconds_of_Use / (Frequency_of_use + 1)
```

The + 1 avoids division by zero when frequency is zero.

3. Train/Test Split

A stratified split was used to preserve the target distribution:

|Split               |Records|
|--------------------|------:|
|Training set        |2,257  |
|Test set            |565    |
|Total after cleaning|2,822  |

The notebook also checked for overlapping feature rows between the training and test sets; no overlap was found in that check.

4. Feature Review

A feature ablation experiment compared Random Forest performance with and without Status. The final tuned model excludes Status.

🤖 Models and Methodology

Three classification models were evaluated using stratified 5-fold cross-validation.

|Model              |Role                        |
|-------------------|----------------------------|
|Logistic Regression|Linear baseline model       |
|Random Forest      |Tree-based ensemble model   |
|Gradient Boosting  |Sequential boosting ensemble|

Hyperparameter Tuning

RandomizedSearchCV was used to tune the Random Forest pipeline without the Status feature. The search evaluated 20 sampled parameter combinations using stratified 5-fold cross-validation and F1-score as the selection metric.

Best parameters found:

|Parameter                     |Selected value|
|------------------------------|-------------:|
|`n_estimators`                |`300`         |
|`max_depth`                   |`30`          |
|`min_samples_split`           |`5`           |
|`min_samples_leaf`            |`2`           |
|`max_features`                |`sqrt`        |
|Best cross-validation F1-score|`0.8618`      |

📈 Model Comparison

Mean cross-validation scores from the notebook:

|Model                                 |Accuracy|Precision|Recall|F1-score|ROC-AUC|PR-AUC|
|--------------------------------------|-------:|--------:|-----:|-------:|------:|-----:|
|Logistic Regression                   |0.8343  |0.4778   |0.8641|0.6152  |0.9290 |0.7454|
|Random Forest                         |0.9637  |0.9431   |0.8122|0.8725  |0.9904 |0.9561|
|Gradient Boosting                     |0.9522  |0.9083   |0.7659|0.8306  |0.9821 |0.9273|
|Tuned Random Forest (without `Status`)|0.9592  |0.8948   |0.8324|0.8618  |0.9860 |0.9234|

The first three rows show the initial model comparison. The tuned Random Forest row reports the separate cross-validation evaluation of the selected tuned estimator without Status.

🏆 Final Model Evaluation

The tuned Random Forest model was evaluated on the held-out test set of 565 records.

Test Set Metrics

|Metric   |Score |
|---------|-----:|
|Accuracy |94.87%|
|Precision|83.53%|
|Recall   |82.56%|
|F1-score |83.04%|
|PR-AUC   |91.33%|

Confusion Matrix

|Actual / Predicted|Stayed (`0`)|Churned (`1`)|
|------------------|-----------:|------------:|
|Stayed (`0`)      |465         |14           |
|Churned (`1`)     |15          |71           |

What These Results Mean

• Precision (83.53%): Of the customers predicted to churn, about 83.53% actually churned.
• Recall (82.56%): The model identified 71 of the 86 customers who churned in the test set.
• F1-score (83.04%): Balances precision and recall.
• PR-AUC (91.33%): Summarizes the precision-recall trade-off across thresholds and is useful for an imbalanced classification problem.

The model’s predictions are estimates and should be interpreted alongside business context and ongoing validation.

🏗️ System Architecture

The project consists of three main parts:

|Component       |Responsibility                                                                             |
|----------------|-------------------------------------------------------------------------------------------|
|Jupyter Notebook|EDA, preprocessing, feature engineering, training, tuning, and evaluation                  |
|FastAPI         |Serves prediction endpoints and validates incoming request data                            |
|Streamlit       |Provides an interactive interface for entering customer information and viewing predictions|

The selected model pipeline is serialized with Joblib for reuse by the prediction application.

🛠️ Tech Stack

|Area                |Tools              |
|--------------------|-------------------|
|Language            |Python             |
|Data handling       |Pandas, NumPy      |
|Visualization       |Matplotlib, Seaborn|
|Machine Learning    |Scikit-learn       |
|Model persistence   |Joblib             |
|API                 |FastAPI, Uvicorn   |
|User interface      |Streamlit          |
|Notebook environment|Jupyter Notebook   |

📂 Project Structure

```text
customer-churn-prediction-system/
├── API/
│   └── Customer-Churn-Prediction-Starter/
│       └── Customer-Churn-Prediction/
│           ├── main.py
│           ├── app.py
│           ├── requirements.txt
│           ├── .env.example
│           └── .gitignore
├── Customer Churn.csv
├── Customer_Churn_Notebook.ipynb
├── models/
│   └── churnguard_pipeline.joblib
└── README.md
```

The tree illustrates the main files and expected model-artifact location; adjust folder names if your repository layout differs.

🚀 Setup and Usage

1. Clone the repository

```bash
git clone https://github.com/miraesmael66-blip/customer-churn-prediction-system.git
cd customer-churn-prediction-system
```

2. Install dependencies

Open a terminal in the API application directory (the directory containing requirements.txt):

```bash
python -m pip install -r requirements.txt
```

3. Run the FastAPI backend

From the directory containing main.py, run:

```bash
python -m uvicorn main:app --reload
```

Then open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health-check endpoint:

```text
http://127.0.0.1:8000/health
```

4. Run the Streamlit interface

Open a second terminal in the directory containing app.py and run:

```bash
python -m streamlit run app.py
```

Keep the FastAPI backend running while using the Streamlit interface. Ensure the model path configured in the backend points to the saved churnguard_pipeline.joblib file.

⚠️ Limitations and Future Work

• The dataset is relatively small and may not represent every telecom customer population.
• Churn classes are imbalanced; model quality should be assessed using multiple metrics.
• Performance may change on new data, so external validation and monitoring would be needed before production use.
• Future work could include deployment to a cloud platform, monitoring, periodic retraining, and further threshold optimization based on business needs.

👩‍💻 Author

Amira Esmail
Machine Learning | Artificial Intelligence | Data Analysis

🔗 GitHub Profile

────────

⭐ If you find this project useful, feel free to star the repository!
