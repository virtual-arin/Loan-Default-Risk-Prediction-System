import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="💰",
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    text-align:center;
    font-size:42px;
    font-weight:700;
}

.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.metric-card {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:12px;
    text-align:center;
}

div[data-testid="stMetric"] {
    background-color:#f8f9fa;
    padding:15px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    '<div class="big-title">💰 Loan Default Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered loan default risk assessment using Machine Learning</div>',
    unsafe_allow_html=True
)

# ======================================================
# APPLICANT INFORMATION
# ======================================================

st.subheader("👤 Applicant Information")

age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

income = st.number_input(
        "Annual Income ($)",
        min_value=1000.0,
        value=50000.0
    )

credit_score = st.slider(
        "Credit Score",
        300,
        850,
        650
    )


months_employed = st.slider(
        "Months Employed",
        0,
        600,
        60
    )

num_credit_lines = st.slider(
        "Number of Credit Lines",
        1,
        20,
        5
    )

employment_type = st.selectbox(
        "Employment Type",
        ["Full-time",
         "Part-time",
         "Self-employed",
         "Unemployed"]
    )

education = st.selectbox(
        "Education",
        ["Bachelor's",
         "High School",
         "Master's",
         "PhD"]
    )

marital_status = st.selectbox(
        "Marital Status",
        ["Divorced",
         "Married",
         "Single"]
    )

has_dependents = st.selectbox(
        "Has Dependents",
        ["No", "Yes"]
    )

st.divider()

# ======================================================
# LOAN INFORMATION
# ======================================================

st.subheader("🏦 Loan Information")

loan_amount = st.number_input(
        "Loan Amount ($)",
        min_value=1000.0,
        value=20000.0
    )

loan_term = st.selectbox(
        "Loan Term",
        [12, 24, 36, 48, 60, 72, 84]
    )

interest_rate = st.slider(
        "Interest Rate (%)",
        1.0,    
        30.0,
        10.0
    )

dti_ratio = st.slider(
        "Debt-To-Income Ratio",
        0.0,
        1.0,
        0.30
    )

loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Auto",
         "Business",
         "Education",
         "Home",
         "Other"]
    )

has_cosigner = st.selectbox(
        "Has Co-Signer",
        ["No", "Yes"]
    )

has_mortgage = st.selectbox(
        "Has Mortgage",
        ["No", "Yes"]
    )

# ======================================================
# FEATURE ENGINEERING
# ======================================================

loan_to_income = loan_amount / income

income_per_creditline = income / num_credit_lines

employment_stability = months_employed / 12

interest_burden = (
    loan_amount * (interest_rate / 100)
) / income

# ======================================================
# BUILD INPUT
# ======================================================

input_df = pd.DataFrame(
    np.zeros((1, len(feature_columns))),
    columns=feature_columns
)

numeric_data = {
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "Loan_to_Income": loan_to_income,
    "Income_per_CreditLine": income_per_creditline,
    "Employment_Stability": employment_stability,
    "Interest_Burden": interest_burden
}

for feature, value in numeric_data.items():
    if feature in input_df.columns:
        input_df.loc[0, feature] = value

categorical_features = [
    f"Education_{education}",
    f"EmploymentType_{employment_type}",
    f"MaritalStatus_{marital_status}",
    f"HasMortgage_{has_mortgage}",
    f"HasDependents_{has_dependents}",
    f"LoanPurpose_{loan_purpose}",
    f"HasCoSigner_{has_cosigner}"
]

for col in categorical_features:
    if col in input_df.columns:
        input_df.loc[0, col] = 1

st.divider()

# ======================================================
# PREDICTION
# ======================================================

if st.button(
    "🔍 Predict Default Risk",
    use_container_width=True
):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Results")

    if prediction == 1:

        st.error(
            "⚠️ High Risk Customer — Loan default is likely."
        )

    else:

        st.success(
            "✅ Low Risk Customer — Loan default is unlikely."
        )