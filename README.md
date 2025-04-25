# Predicting-Credit-Risk-for-Loan-Applicants

## 1. Introduction

This project presents a machine learning-based application to assess credit risk levels using unsupervised learning. The goal is to predict whether an individual falls into a **low-risk** or **high-risk** group based on demographic and financial data from the **German Credit Dataset**. The solution is deployed through an interactive **Streamlit web application**, allowing real-time predictions based on user input.

---

## 2. Dataset Overview

- **Dataset Name:** German Credit Data
- **Source:** UCI Machine Learning Repository
- **Total Records:** 1,000
- **Features:** 10 selected attributes including:
  - Age
  - Sex
  - Job
  - Housing
  - Saving accounts
  - Checking account
  - Credit amount
  - Duration (loan term)
  - Purpose
  - Risk (target, not used for training)

---

## 3. Methodology

### 3.1 Data Exploration and Preprocessing

- **Missing Values:** Identified and imputed as appropriate (e.g., 'NaN' filled as its own category).
- **Categorical Encoding:** Used `LabelEncoder` to convert categorical variables into numerical representations.
- **Feature Scaling:** Applied `StandardScaler` to normalize numeric features (e.g., age, credit amount).
- **Dimensionality:** All 10 features were retained to maintain full customer profile information.

### 3.2 Model Selection: KMeans Clustering

Since the dataset lacks a clean binary classification label (the "Risk" column is not consistently binary), an **unsupervised approach** was selected.

**KMeans Clustering** was chosen because:
- It is a simple, fast algorithm suitable for well-structured numerical data.
- Useful when there is no definitive label but natural grouping may exist.
- Helps reveal hidden patterns or risk clusters within data.

### 3.3 Model Training

- KMeans model trained with `n_clusters = 2` to identify **two distinct credit risk groups**.
- Fitted on the full preprocessed dataset (excluding the "Risk" label).
- Output clusters interpreted as:
  - **Cluster 0:** Presumed lower-risk group
  - **Cluster 1:** Presumed higher-risk group

### 3.4 Model Deployment

- The model, along with encoders and scaler, was saved as a bundle using `pickle`.
- Streamlit app was created for UI:
  - Accepts user input through form elements (sliders, selectboxes, etc.).
  - Transforms input using trained encoders and scaler.
  - Predicts credit risk group using the trained KMeans model.

---

## 4. Results and Visual Insights

- **Cluster Analysis:**
  - Clear distinction between high and low credit amount borrowers.
  - Individuals with low checking and saving balances tended to fall into the high-risk cluster.

- **User Prediction:**
  - Real-time feedback is given to users as:
    - "You belong to Group 0 – Good Credit Risk"
    - "You belong to Group 1 – Bad Credit Risk"

- **User Interface:**
  - Simple, responsive layout using Streamlit widgets.
  - Handles preprocessing and prediction behind-the-scenes.

---

## 5. Conclusions

### 5.1 Why KMeans Was Used

- The project aimed to reveal underlying credit groupings without depending on a binary label.
- KMeans effectively segmented the dataset into meaningful clusters.
- Provided a fast, unsupervised way to explore risk profiles.

### 5.2 Limitations

- As KMeans is unsupervised, there is no guaranteed label correctness.
- Sensitive to initialization and outliers.
- Assumes spherical clusters and equal variance.

### 5.3 Future Work

- Add **supervised model** support (e.g., RandomForest, XGBoost) using true "Risk" labels.
- Integrate **model explainability** using SHAP or LIME.
- Enable **database storage** for user predictions.
- Deploy the app to **Streamlit Cloud or Hugging Face Spaces**.

---

## 6. Appendix

### 6.1 Technologies Used
- Python 3.10
- Pandas, NumPy
- Scikit-learn (KMeans, preprocessing)
- Streamlit
- Pickle (for model serialization)

### 6.2 Installation & Run Instructions
```bash
pip install streamlit scikit-learn pandas numpy
streamlit run credit_risk_predictor_v2.py
```

---

**Project by:** Vaibhav Bansal
**Last Updated:** April 2025

