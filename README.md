Customer Churn Prediction System

1. Project Overview

The Customer Churn Prediction System is a Machine Learning project that predicts whether a customer is likely to leave a telecommunications service.

The project uses customer information, subscription details, and usage behavior to identify customers who may be at risk of churn.

It includes a trained Machine Learning model, a FastAPI backend for predictions, and a Streamlit interface for user interaction.

2. Project Objectives

* Analyze customer data and identify patterns related to churn.
* Perform data cleaning and preprocessing.
* Engineer useful features to improve prediction.
* Train and evaluate multiple Machine Learning models.
* Handle class imbalance and evaluate model performance.
* Provide predictions through a simple interactive interface.

3. Dataset

The dataset contains customer subscription, usage, demographic, and churn information.

Original dataset:

* Number of records: 3,150
* Number of columns: 14
* Target variable: Churn
* Class distribution: 495 churned customers and 2,655 non-churned customers.

The dataset contains an imbalanced target, with approximately 15.71% of customers belonging to the churn class.

4. Data Preprocessing and Feature Engineering

The preprocessing workflow includes:

* Exploring the dataset and analyzing feature distributions.
* Checking missing values and duplicate records.
* Removing duplicate records.
* Handling conflicting records with identical features but different target labels.
* Analyzing relationships between customer behavior and churn.
* Creating the Avg_Usage_Duration feature:

Avg_Usage_Duration = Seconds of Use / (Frequency of use + 1)

* Splitting the data into training and testing sets while preserving the target class distribution.

After preprocessing, 2,822 records were retained.

5. Machine Learning Models

Several models were evaluated during experimentation, including:

* Logistic Regression
* Random Forest
* Gradient Boosting

Random Forest was selected for the final model after comparing performance and tuning hyperparameters.

The final model excludes the Status feature to avoid relying on a potentially problematic or overly predictive feature.

Hyperparameter tuning was performed using RandomizedSearchCV with stratified cross-validation.

The selected hyperparameters were:

* n_estimators = 300
* max_depth = 30
* min_samples_split = 5
* min_samples_leaf = 2
* max_features = 'sqrt'

6. Final Model Evaluation

The final model was evaluated on a held-out test set.

Metric	Result
Accuracy	Approximately 94.87%
Precision	83.53%
Recall	82.56%
F1-score	83.04%
PR-AUC	91.33%

Confusion Matrix

	Predicted No Churn	Predicted Churn
Actual No Churn	465	14
Actual Churn	15	71

The model correctly identified 71 churned customers out of 86 actual churn cases in the test set.

Precision, recall, and F1-score are included because accuracy alone may not adequately describe performance on an imbalanced dataset.

7. System Architecture

The project consists of three main components:

Jupyter Notebook

Used for data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

FastAPI Backend

Loads the saved model, validates customer inputs, calculates the engineered feature, and returns prediction results.

Streamlit Frontend

Provides an interactive form where users enter customer information and view the predicted churn class and estimated churn probability.

8. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Streamlit
* Requests
* Jupyter Notebook

9. Project Structure

final pro/
├── Customer Churn.csv
├── Customer_Churn_Notebook.ipynb
├── README.md
└── API/
    ├── artifacts/
    │   └── churnguard_pipeline.joblib
    └── Customer-Churn-Prediction-Starter/
        └── Customer-Churn-Prediction/
            ├── main.py
            ├── app.py
            └── requirements.txt

10. How to Run the Project

Step 1: Install Dependencies

Open a terminal in the application directory:

cd API/Customer-Churn-Prediction-Starter/Customer-Churn-Prediction

Install the required packages:

pip install -r requirements.txt

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

Open a second terminal in the same application directory and run:

python -m streamlit run app.py

Streamlit will display the application URL in the terminal.

Keep the FastAPI backend running while using the interface.

11. Limitations

* Predictions depend on the quality and representativeness of the training dataset.
* Model performance may change when applied to new customer populations.
* The estimated churn probability is not a guarantee of future customer behavior.
* The model should be monitored and evaluated before being used for real business decisions.

12. Future Improvements

* Add model monitoring and performance tracking.
* Improve explainability using feature importance or SHAP.
* Add more robust input validation.
* Evaluate decision thresholds based on business requirements.
* Deploy the application to a cloud platform.
* Integrate automated testing and a reproducible training pipeline.

13. Conclusion

This project demonstrates an end-to-end Machine Learning workflow, from data exploration and preprocessing to model training, evaluation, API development, and interactive deployment.

It provides a practical example of applying Machine Learning to customer churn prediction.