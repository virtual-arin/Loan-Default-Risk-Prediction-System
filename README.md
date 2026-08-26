# Loan Default Risk Prediction System 💵

## 🏦 Industry Domain 

Banking & Financial Services

-----
 
## 🏢 About the Company

Langford Banking Group is a banking and financial services company focused on providing safe, reliable, and data-driven lending solutions. The bank uses technology and analytics to assess borrower risk, improve credit decisions, reduce financial losses, and support responsible lending. Langford Banking Group aims to make loan approvals more efficient while strengthening credit risk management across its lending operations and customer base.

-----
 
## 🤔 Problem Statement

Langford Banking Group wants to predict whether a borrower is likely to default on a loan before approval. The model uses demographic, financial, employment, credit, and loan-related information to identify high-risk borrowers and help the bank make safer, data-driven lending decisions while minimizing avoidable credit risk for lenders.

-----

## 🎯 Objective

Build a classification model that predicts loan default risk using borrower demographic, financial, employment, credit, and loan information. The goal is to identify high-risk borrowers, reduce financial losses, improve lending decisions, and strengthen credit risk management.

-----

## 🥸 Dataset Overview

[Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

* The dataset contains borrower information such as income, employment details, credit history, loan characteristics, and repayment behavior. The **Target Variable:** is `Default` where, 0 → No Default and 1 → Default

-----

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-Learn
* Joblib
* Streamlit

-----

## 📊 Data Visualization

1. **Categorical Feature Univariate Distribution**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/categorical_feature_univariate_distributions.png" width="100%">

2. **Numerical Feature Univariate Distributions**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/numerical_feature_univariate_distributions.png" width="100%">

3. **Categorical Feature Bivariate Analysis**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/categorical_feature_distributions_by_default.png" width="100%">

4. **Numerical Feature Bivariate Analysis**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/numerical_feature_distributions_by_default.png" width="100%">

5. **What is the overall default rate?**
- **About 88.4% of customers have fully paid the loan while 11.6% customers have defaulted the loan.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/default_distribution.png" width="100%">

6. **What is the overall default rate (visualize as pie chart)?**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/default_pie_visualization.png" width="100%">

7. **Do borrowers with lower credit scores default more?**
- **Credit scores look nearly identical for both successful payers and those who default on their loans.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/credit_score_distribution.png" width="100%">

8. **Does debt-to-income ratio (DTI) increase default risk?**
- **A borrower's debt to income ratio shows almost no meaningful relationship between those who pay and those who default.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/dti_distribution.png" width="100%">

9. **Are larger loans more likely to default?**
- **Borrowers who are failing to repay their loans tend to request slightly larger loan amounts on average overall.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/loan_amount_distribution.png" width="100%">

10. **How does income relate to default?**
- **People who default on their loans generally earn noticeably lower annual incomes compared to successful, reliable loan payers.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/income_distribution.png" width="100%">

11. **Does employment stability matter?**
- **Borrower with less employment month are tends to default more than borrower with stable employment.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/employment_stability.png" width="100%">

12. **Which employment types are riskiest?**
- **Unemployed individuals show the highest risk of defaulting, while full-time workers consistently show the lowest overall risk.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/employment_types.png" width="100%">

13. **Does education level affect default?**
- **Borrowers holding only a high school diploma represent the highest risk group for failing to repay their loans.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/education_default.png" width="100%">

14. **Which loan purposes have the highest risk?**
- **Loans taken out for business purposes carry the highest default risk, while home loans remain the safest.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/default_by_loan_purpose.png" width="100%">

15. **What variables are most correlated with default?**
- **Most numerical features show weak relationships, with borrower age and interest rates only slightly affecting loan defaults.**
<img src="https://github.com/virtual-arin/Loan-Default-Risk-Prediction-System/blob/main/images/correlation_heatmap.png">

-----

## 🔄 Workflow

### 1. Data Analysis

* Performed EDA to understand borrower behavior.
* Analyzed feature distributions and default patterns.

### 2. Feature Engineering

* Created important default indicating features like Loan_to_Income, Income_per_CreditLine, Employment_Stability and Interest_Burden
* Prepared data for machine learning models.

### 3. Model Selection

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### 4. Choosing Metric

In this imbalanced loan-default problem, I prioritized recall because missing a true defaulter is more expensive than reviewing a few additional false-positive cases because missing a defaulter can lead to financial loss.

### 5. Hyperparameter Optimization

* Used GridSearchCV to improve model performance.
* Selected the best-performing model based on validation metrics.

### 6. Deployment

* Saved the trained model using Joblib.
* Built a Streamlit application for real-time loan default prediction.

## 📈 Baseline Summary

| Model | Precision | Recall | F1-score | Selected Model | 
|---|---:|---:|---:|---| 
| **Logistic Regression** | 0.23 | **0.70** | **0.34** | **🏆 Champion (Best for detecting defaulters)** | 
| Decision Tree | 0.19 | 0.19 | 0.19 | Poor overall | 
| Random Forest | **0.44** | 0.17 | 0.25 | Better precision, poor recall | 
| Gradient Boosting | **0.61** | 0.06 | 0.11 | Highest precision, worst recall |

---

## 🏆 After Hyperparameter Optimization

| Model | Recall (Class 1) | Status | 
|---|---:|---| 
| **Logistic Regression** | **68.6%** | 🏆 **Champion Model** | 
| Random Forest | 63.0% | Good Performance | 
| Decision Tree | 62.7% | Good Performance | 
| Gradient Boosting | 7.6% | Poor Performance |

-----

## 🚀 Business Impact

* Identifies high-risk borrowers before loan approval.
* Helps reduce potential credit losses.
* Improves data-driven lending decisions.
* Detects ~69% of actual defaulters using logistic regression model.
* Supports automated and scalable credit risk assessment.
* Strengthens the bank's overall credit risk management.
