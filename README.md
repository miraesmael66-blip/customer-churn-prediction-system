📊 ChurnGuard AI | Customer Churn Prediction System

An End-to-End Machine Learning Project

An end-to-end Machine Learning system designed to predict customer churn in the telecommunications industry using customer demographics, subscription information, and usage behavior.

The project combines data analysis, feature engineering, model comparison, hyperparameter tuning, and model evaluation with a FastAPI backend and Streamlit interface to deliver an interactive customer churn prediction system.

⸻

🎯 Project Overview

Customer churn occurs when customers discontinue a service or switch to another provider. Predicting churn helps businesses identify customers at risk and make better retention decisions.

ChurnGuard AI uses Machine Learning classification models to identify potential churners based on customer data.

✨ Key Highlights

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data Cleaning and Preprocessing
* ⚙️ Feature Engineering
* 🤖 Training and Comparison of Multiple ML Models
* 🔍 Hyperparameter Tuning using RandomizedSearchCV
* 📈 Evaluation using Multiple Classification Metrics
* 💾 Model Serialization using Joblib
* 🚀 REST API Development using FastAPI
* 🖥️ Interactive Prediction Interface using Streamlit

⸻

📂 Dataset Overview

The dataset contains telecommunications customer information, including subscription details, usage behavior, demographic attributes, and churn labels.

Dataset Information	Value
Original Records	3,150
Original Features	14
Target Variable	Churn
Churned Customers	495
Non-Churned Customers	2,655
Churn Rate	~15.71%
Records Retained After Preprocessing	2,822

⚠️ Class Imbalance

The dataset contains fewer churned customers than non-churned customers.

Therefore, accuracy alone is insufficient to evaluate the model. Precision, Recall, F1-score, and PR-AUC are also considered to provide a more informative evaluation.

⸻

🔬 Data Preprocessing & Feature Engineering

The preprocessing workflow focuses on improving data quality and preparing meaningful features for Machine Learning.

🧹 Data Cleaning

* Explored feature distributions and relationships.
* Checked for missing values and duplicate records.
* Removed duplicate records.
* Handled records with identical feature values but conflicting target labels.

⚙️ Feature Engineering

Created a new feature called Avg_Usage_Duration to represent average usage duration per usage frequency.

Formula:

Avg_Usage_Duration = Seconds of Use / (Frequency of use + 1)

📌 Data Splitting

The dataset was split into training and testing sets using stratification to preserve the target-class distribution.

After preprocessing, 2,822 records were retained.

⸻

🤖 Machine Learning Models

Three classification algorithms were evaluated to identify the most suitable model for the churn prediction task.

Model	Description
Logistic Regression	Linear classification baseline
Random Forest	Ensemble of decision trees
Gradient Boosting	Sequential boosting-based ensemble

🏆 Final Model Selection

Random Forest was selected following model comparison and hyperparameter tuning.

The Status feature was excluded from the final model to avoid relying on a potentially problematic or overly predictive feature.

🔍 Hyperparameter Optimization

RandomizedSearchCV was used with stratified cross-validation to search for a suitable hyperparameter configuration.

Selected Hyperparameters

Parameter	Value
n_estimators	300
max_depth	30
min_samples_split	5
min_samples_leaf	2
max_features	sqrt

⸻

📈 Final Model Performance

The final Random Forest model was evaluated on a held-out test set.

📊 Evaluation Metrics

Metric	Score
Accuracy	94.87%
Precision	83.53%
Recall	82.56%
F1-score	83.04%
PR-AUC	91.33%

🔎 Confusion Matrix

Actual / Predicted	No Churn	Churn
Actual No Churn	465	14
Actual Churn	15	71

💡 Interpreting the Results

* Accuracy: Measures the proportion of all test predictions that were correct.
* Precision: Measures how many customers predicted to churn actually churned.
* Recall: Measures how many actual churners were correctly identified.
* F1-score: Balances Precision and Recall.
* PR-AUC: Summarizes the precision-recall trade-off across classification thresholds.

The model correctly identified 71 out of 86 customers who churned in the test set.

Using multiple evaluation metrics provides a more complete understanding of model performance than accuracy alone, especially when working with imbalanced data.

⸻

🏗️ System Architecture

ChurnGuard AI combines three main components to support the complete Machine Learning workflow.

1. 📓 Jupyter Notebook — ML Pipeline

Responsible for:

* Exploratory Data Analysis
* Data Cleaning and Preprocessing
* Feature Engineering
* Model Training and Comparison
* Hyperparameter Tuning
* Model Evaluation

Main file: Customer_Churn_Notebook.ipynb

2. ⚡ FastAPI — Backend

The API provides the prediction functionality.

It is responsible for:

* Loading the trained Machine Learning model.
* Validating customer inputs.
* Calculating the engineered feature.
* Returning churn predictions and estimated churn probabilities.

3. 🖥️ Streamlit — Frontend

The interactive interface allows users to:

* Enter customer information.
* Submit customer data for prediction.
* View the prediction results.

The Streamlit application communicates with the FastAPI backend.

⸻

🛠️ Technologies & Tools

Category	Technologies
Programming Language	Python
Data Manipulation	Pandas, NumPy
Data Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Model Serialization	Joblib
Backend API	FastAPI
API Server	Uvicorn
Frontend Interface	Streamlit
HTTP Requests	Requests
Development Environment	Jupyter Notebook

⸻

📁 Project Structure

customer-churn-prediction-system/
│
├── API/
│   └── Customer-Churn-Prediction-Starter/
│       └── Customer-Churn-Prediction/
│           ├── main.py
│           ├── app.py
│           ├── requirements.txt
│           ├── .gitignore
│           └── .env.example
│
├── Customer Churn.csv
├── Customer_Churn_Notebook.ipynb
├── churnguard_pipeline.joblib
└── README.md

📌 Main Components

File	Purpose
Customer Churn.csv	Original dataset
Customer_Churn_Notebook.ipynb	Analysis, training, and evaluation
churnguard_pipeline.joblib	Serialized trained model pipeline
main.py	FastAPI backend
app.py	Streamlit frontend
requirements.txt	Python dependencies
README.md	Project documentation

⸻

🚀 Installation & Setup

Follow these steps to run the project locally.

Prerequisites

* Python installed
* pip package manager
* Git

1️⃣ Clone the Repository

git clone https://github.com/miraesmael66-blip/customer-churn-prediction-system.git

Navigate to the project directory:

cd customer-churn-prediction-system

2️⃣ Install Dependencies

Navigate to the API application directory:

cd API/Customer-Churn-Prediction-Starter/Customer-Churn-Prediction

Install the required packages:

python -m pip install -r requirements.txt

3️⃣ Start the FastAPI Backend

Run the following command from the API application directory:

python -m uvicorn main:app --reload

The API will be available at:

http://127.0.0.1:8000

Available API Pages

Endpoint	Purpose
/docs	Interactive API documentation
/health	API health check

Open the interactive documentation at:

http://127.0.0.1:8000/docs

Check the API health at:

http://127.0.0.1:8000/health

4️⃣ Start the Streamlit Interface

Open a second terminal and navigate to the same API application directory.

Run:

python -m streamlit run app.py

Open the URL displayed in the terminal to access the interface.

Important: Keep the FastAPI backend running while using the Streamlit application.

⚠️ Model File Path

The trained model file, churnguard_pipeline.joblib, is currently stored in the repository root.

Ensure that the model-loading path in main.py correctly points to the file’s actual location when running the project from a fresh clone.

⸻

⚠️ Limitations

* Predictions depend on the quality and representativeness of the training data.
* Model performance may vary across different customer populations.
* Predicted probabilities are estimates, not guarantees of future customer behavior.
* Further monitoring and evaluation are necessary before using the model for real business decisions.

⸻

🔮 Future Improvements

Potential future enhancements include:

* 📊 Model monitoring and performance tracking.
* 🔄 Periodic model retraining with updated customer data.
* ☁️ Cloud deployment for broader accessibility.
* 📈 Additional analytics to support customer retention strategies.

⸻

👩‍💻 Author

Amira Esmail

Machine Learning | Artificial Intelligence | Data Analysis

🔗 GitHub Profile

⸻

⭐ If you find this project interesting, consider giving the repository a star!
