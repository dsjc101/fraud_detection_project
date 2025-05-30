import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
import joblib

# Load trained model only
model = joblib.load('C:\\Users\\divya\\OneDrive\\Desktop\\fraud_detection_project\\models\\xgb_fraud_final_model.pkl')

# Fraud threshold
threshold = 0.05

st.title("💳 Credit Card Fraud Detection")
st.write("Upload raw transaction data. The app will preprocess it and flag suspicious transactions.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def preprocess_input(df):
    df = df.copy()

    # Extract Hour and Night flag
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Is_Night_Transaction'] = df['Hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)

    # Apply QuantileTransformer inline
    exclude = ['Class', 'Is_Night_Transaction']
    numeric_cols = df.select_dtypes(include='number').columns
    features_to_transform = [col for col in numeric_cols if col not in exclude]

    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    df[features_to_transform] = qt.fit_transform(df[features_to_transform])

    # Drop 'Time'
    df = df.drop(columns=['Time'])

    return df

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write("### Raw Input Preview", df_raw.head())

    df_processed = preprocess_input(df_raw)
    st.write("### Preprocessed Input Preview", df_processed.head())

    # Prediction
    X = df_processed.drop(columns=['Class'], errors='ignore')
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    df_raw['Fraud_Probability'] = proba
    df_raw['Fraud_Prediction'] = preds
    
    total_txns = len(df_raw)
    total_frauds = df_raw['Fraud_Prediction'].sum()

    st.markdown(f"**🧾 Total Transactions:** `{total_txns}`")
    st.markdown(f"**🚨 Total Predicted Frauds:** `{total_frauds}`")


    st.write(" Prediction Results")
    st.dataframe(df_raw[['Fraud_Probability', 'Fraud_Prediction']])

    frauds = df_raw[df_raw['Fraud_Prediction'] == 1]

    if not frauds.empty:
        st.markdown(" Likely Fraudulent Transactions")
        st.dataframe(frauds)
        csv = frauds.to_csv(index=False).encode()
        st.download_button(" Download Likely Frauds CSV", csv, "likely_frauds.csv", "text/csv")
    else:
        st.success(" No likely frauds detected in this batch.")
