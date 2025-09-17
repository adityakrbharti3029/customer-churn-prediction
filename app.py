import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
try:
    pipeline = joblib.load('churn_model_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file ('churn_model_pipeline.pkl') not found. Please run 'python model.py' first to train and save the model.")
    st.stop()

st.title('Customer Churn Prediction')

# --- Sidebar for User Input (All original features) ---
st.sidebar.header('Customer Details')

gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
senior_citizen = st.sidebar.selectbox('Senior Citizen', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 24)
phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
# Using realistic ranges from the original dataset
monthly_charges = st.sidebar.slider('Monthly Charges', 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider('Total Charges', 18.0, 8700.0, 1500.0)

# --- Prediction Button and Output ---
if st.sidebar.button('Predict'):
    
    input_data = pd.DataFrame({
        'gender': [gender], 'SeniorCitizen': [senior_citizen], 'Partner': [partner],
        'Dependents': [dependents], 'tenure': [tenure], 'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines], 'InternetService': [internet_service],
        'OnlineSecurity': [online_security], 'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection], 'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv], 'StreamingMovies': [streaming_movies],
        'Contract': [contract], 'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method], 'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    prediction = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)

    st.subheader('Prediction Result')

    if prediction[0] == 1:
        st.error(f'This customer is LIKELY to churn.')
        st.write(f'**Churn Probability:** {prediction_proba[0][1]*100:.2f}%')
    else:
        st.success(f'This customer is LIKELY to stay.')
        st.write(f'**Loyalty Probability:** {prediction_proba[0][0]*100:.2f}%')