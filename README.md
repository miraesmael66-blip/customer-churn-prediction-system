Customer Churn Prediction System

1. Project Overview

The Customer Churn Prediction System is an end-to-end Machine Learning project that predicts whether a telecommunications customer is likely to leave the service.

The project analyzes customer subscription details, usage behavior, and demographic information to identify customers who may be at risk of churn.

It includes data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, model evaluation, a FastAPI backend, and a Streamlit interface.

2. Project Objectives

* Explore customer data and identify patterns associated with churn.
* Clean and preprocess the dataset.
* Engineer useful features for prediction.
* Train and compare multiple Machine Learning models.
* Tune the selected model using cross-validation.
* Evaluate performance using metrics suitable for imbalanced classification.
* Save the trained model for reuse.
* Provide predictions through a FastAPI endpoint and an interactive Streamlit interface.

3. Dataset

The dataset contains customer subscription, usage, demographic, and churn information.

* Original records: 3,150
* Original columns: 14
* Target variable: Churn
* Churned customers: 495
* Non-churned customers: 2,655
* Churn rate: approximately 15.71%

The dataset has an imbalanced target distribution, so accuracy alone is not sufficient to evaluate the model.

4. Data Preprocessing and Feature Engineering

The workflow includes:

* Exploring feature distributions and relationships.
* Checking missing values and duplicate records.
* Removing duplicate records.
* Handling records with identical feature values but conflicting target labels.
* Creating the Avg_Usage_Duration feature:

Avg_Usage_Duration = Seconds of Use / (Frequency of use + 1)

* Splitting the data into training and testing sets using stratification.

After preprocessing, 2,822 records were retained.

5. Machine Learning Models

Three classification models were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

Random Forest was selected after model comparison and hyperparameter tuning.

The Status feature was excluded from the final model to avoid relying on a potentially problematic or overly predictive feature.

Hyperparameter tuning was performed using RandomizedSearchCV with stratified cross-validation.

Selected Hyperparameters

* n_estimators: 300
* max_depth: 30
* min_samples_split: 5
* min_samples_leaf: 2
* max_features: sqrt

6. Final Model Evaluation

The final model was evaluated on a held-out test set.

Metric	Result
Accuracy	94.87%
Precision	83.53%
Recall	82.56%
F1-score	83.04%
PR-AUC	91.33%

Confusion Matrix

Actual / Predicted	No Churn	Churn
Actual No Churn	465	14
Actual Churn	15	71

The model correctly identified 71 of the 86 customers who churned in the test set.

Precision, recall, F1-score, and PR-AUC provide additional insight into performance on the imbalanced dataset.

7. System Architecture

Jupyter Notebook

Used for data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

FastAPI Backend

Loads the trained model, validates customer inputs, calculates the engineered feature, and returns churn predictions and estimated churn probabilities.

Streamlit Frontend

Provides an interactive form for entering customer information and viewing prediction results.

8. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Streamlit
* Requests
* Jupyter Notebook

9. Project Structure

The repository contains the dataset, notebook, saved model, and API application.

customer-churn-prediction-system/
├── API/
│   └── Customer-Churn-Prediction-Starter/
│       └── Customer-Churn-Prediction/
│           ├── main.py
│           ├── app.py
│           ├── requirements.txt
│           ├── .gitignore
│           └── .env.example
├── Customer Churn.csv
├── Customer_Churn_Notebook.ipynb
├── README.md
└── churnguard_pipeline.joblib

Note: The saved model is currently stored in the repository root. The API’s model-loading path must point to this location, or the model must be moved to the path expected by the code.

10. How to Run the Project Locally

Step 1: Install Dependencies

Open a terminal in the API application directory:

cd API/Customer-Churn-Prediction-Starter/Customer-Churn-Prediction

Install the required packages:

python -m pip install -r requirements.txt

Step 2: Start the FastAPI Backend

Run:

python -m uvicorn main:app --reload

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

Step 3: Start the Streamlit Interface

Open a second terminal in the same API application directory and run:

python -m streamlit run app.py

Use the URL printed in the terminal to open the interface.

Keep the FastAPI backend running while using Streamlit.

Important Model Path Note

The API must be able to locate churnguard_pipeline.joblib when it starts.

Before running the project from a fresh clone, verify that the model-loading path in main.py matches the model’s actual location in the repository. The model file is currently stored in the repository root.

11. Limitations

* Predictions depend on the quality and representativeness of the training data.
* Performance may vary when the model is applied to different customer populations.
* Predicted probabilities are estimates, not guarantees of future customer behavior.
* The model should be monitored and evaluated before being used for real business decisions.

12. Future Improvements

* Add model monitoring and performance tracking.
* Improve explainability using feature importance or SHAP.
* Expand automated testing.
* Evaluate decision thresholds based on business requirements.
* Deploy the application to a cloud platform.
* Build a reproducible model-training pipeline.

13. Conclusion

This project demonstrates an end-to-end Machine Learning workflow, from data exploration and preprocessing to model comparison, evaluation, model saving, API development, and an interactive user interface.

It provides a practical example of applying Machine Learning to customer churn prediction.
