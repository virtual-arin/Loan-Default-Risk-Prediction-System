# Loan Default Risk Prediction

## 🎉 Problem Statement
A bank provides various types of loans (home loans, personal loans, etc.) to customers. However, some borrowers fail to repay their loans (default), which results in significant financial losses.

## 📌 Overview
Built a machine learning model to predict whether a loan applicant will default, helping reduce financial risk in banking.

## 🎯 Objective
- Classify applicants as **Default (1)** or **Not Default (0)**
- Improve loan approval decisions and minimize losses

## 📊 Dataset
- 255,347 records, 18 features  
- Includes demographics, financial profile, and loan details  
- Default rate: ~11.6% (imbalanced dataset)

## 🛠️ Tech Stack
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

## ⚙️ Approach
- Performed EDA and feature analysis  
- Preprocessed data (handled imbalance, removed irrelevant features)  
- Trained **Logistic Regression** (`class_weight='balanced'`)  

## 📈 Results
- Accuracy: **67%**  
- Recall (Default): **69%**  
- Insight: A high recall means the model catches most potential defaulters.

## 🚀 Run Locally
```bash
pip install -r requirements.txt