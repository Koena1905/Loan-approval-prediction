import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===== Load Model and Scaler =====
with open("loan_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("loan_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===== Page Config =====
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

# ===== Custom Styling =====
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton button {
        background-color: #2ecc71;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏦 Loan Approval Prediction")

st.write("### Fill in the details below to check your loan eligibility:")

# ===== Collect Inputs =====
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    credit_history = st.selectbox("Credit History", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, max_value=1000000, step=1000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, max_value=1000000, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, max_value=1000, step=1)
    loan_term = st.number_input("Loan Amount Term (in days)", min_value=0, max_value=500, step=10)

# ===== Encoding Maps =====
gender_map = {"Male": 1, "Female": 0}
married_map = {"No": 0, "Yes": 1}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_map = {"No": 0, "Yes": 1}
credit_map = {"No": 0, "Yes": 1}
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

# ===== Prepare Feature Array =====
features = np.array([[
    gender_map[gender],
    married_map[married],
    dependents_map[dependents],
    education_map[education],
    self_map[self_employed],
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_map[credit_history],
    property_map[property_area]
]])

# Scale features
features_scaled = scaler.transform(features)

# ===== Prediction =====
if st.button("Predict Loan Eligibility"):
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]

    if prediction[0] == 1:
        st.success(f"🎉 Congratulations! You are eligible for the loan.")
    else:
        st.error(f"❌ Unfortunately, you are not eligible for the loan.")

    # Show prediction probabilities
    st.write("### Prediction Confidence")
    st.write(f"Eligible: {probabilities[1]*100:.2f}%")
    st.write(f"Not Eligible: {probabilities[0]*100:.2f}%")

    # ===== Plot probability chart =====
    fig, ax = plt.subplots()
    labels = ["Not Eligible", "Eligible"]
    ax.bar(labels, probabilities, color=['red', 'green'])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)