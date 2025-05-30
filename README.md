Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions. It includes a trained XGBoost model and a Streamlit dashboard for interactive fraud detection on uploaded transaction data.

 Project Structure
- data/: Raw and preprocessed data
- notebooks/: EDA, modeling, feature engineering
- models/: Saved trained model (`xgb_model.pkl`)
- app/: Streamlit app (`fraud_app.py`)

 How to Run the App
1. Install dependencies:
pip install -r requirements.txt
2. Run the Streamlit app:
streamlit run app/fraud_app.py
3. Upload a CSV file and get fraud predictions with probability scores.

 Features
- Handles raw data preprocessing
- XGBoost model with class imbalance handling
- Shows fraud probabilities and predictions
- Downloadable list of suspicious transactions
