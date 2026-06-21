# Loan Default Risk Prediction System 💳📊

## 🏦 Business Domain

Banking & Financial Services
 
## 🤔 Problem Statement

Loan defaults can lead to significant financial losses for lenders. Accurately identifying high-risk borrowers before loan approval helps reduce credit risk and improve lending decisions.

## 🎯 Project Objective

Build a machine learning classification model to predict whether a borrower is likely to default on a loan based on demographic, financial, and credit-related information.

## 📊 Dataset Overview

The dataset contains borrower information such as income, employment details, credit history, loan characteristics, and repayment behavior.

**Target Variable:** `Default`

* 0 → No Default
* 1 → Default

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn
* Joblib
* Streamlit

## 📂 Project Structure

```text
├── data/
├── notebooks/
│   ├── Data Analysis.ipynb
│   ├── Feature Engineering.ipynb
│   └── Model Training & Evaluation.ipynb
├── models/
│   └── loan_default_risk_model.pkl
├── app.py
└── README.md
```

## 🔄 Workflow

### 1. Data Analysis

* Performed EDA to understand borrower behavior.
* Analyzed feature distributions and default patterns.
* Handled missing values and outliers.

### 2. Feature Engineering

* Encoded categorical features.
* Created financial and credit-related indicators.
* Prepared data for machine learning models.

### 3. Model Training

Trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### 4. Hyperparameter Optimization

* Used GridSearchCV to improve model performance.
* Selected the best-performing model based on validation metrics.

### 5. Deployment

* Saved the trained model using Joblib.
* Built a Streamlit application for real-time loan default prediction.

## 📈 Results Summary

| Model               | Status             |
| ------------------- | ------------------ |
| Gradient Boosting   | 🏆 Champion Model  |
| Adaboost            | Strong Performance |
| Random Forest	      | Moderate           |
| Logistic Regression | Baseline           |

## 🚀 Business Impact

* Reduces loan default risk.
* Improves lending decisions.
* Supports automated credit risk assessment.
* Enhances financial risk management.