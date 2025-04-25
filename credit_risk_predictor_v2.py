
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and preprocessing tools
with open("credit_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

kmeans = bundle["kmeans_model"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]
features = bundle["features"]

st.title("Credit Risk Group Prediction")
st.markdown("This app predicts your credit risk group based on KMeans clustering of the German Credit dataset.")

st.header("Enter Your Credit Information")

# Collect user input
age = st.slider("Age", 18, 80, 30)
sex = st.selectbox("Sex", label_encoders['Sex'].classes_.tolist())
job = st.selectbox("Job (0 = Unskilled, 3 = Highly Skilled)", [0, 1, 2, 3])
housing = st.selectbox("Housing", label_encoders['Housing'].classes_.tolist())
saving_accounts = st.selectbox("Saving Accounts", label_encoders['Saving accounts'].classes_.tolist())
checking_account = st.selectbox("Checking Account", label_encoders['Checking account'].classes_.tolist())
credit_amount = st.number_input("Credit Amount", min_value=0, max_value=20000, value=1000)
duration = st.slider("Duration (months)", 4, 72, 24)
purpose = st.selectbox("Purpose", label_encoders['Purpose'].classes_.tolist())

# Create dataframe from input
user_input = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": purpose
}])

# Apply label encoding
for col in label_encoders:
    user_input[col] = label_encoders[col].transform(user_input[col])

# Scale features
user_scaled = scaler.transform(user_input[features])

# Predict the cluster
prediction = kmeans.predict(user_scaled)[0]

st.subheader("Predicted Credit Risk Group")
if prediction == 0:
    st.success("Good Credit Risk")
else:
    st.warning("Bad Credit Risk")
