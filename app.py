import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load trained model
model = load_model('churn_model.h5')

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📱 Telco Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of churn.")

# --- Input Fields ---
tenure = st.slider('Tenure (Months)', min_value=0, max_value=72, value=12)
monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, value=70.0, step=1.0)
total_charges = st.number_input('Total Charges ($)', min_value=0.0, value=1400.0, step=10.0)

contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

gender = st.selectbox('Gender', ['Male', 'Female'])
partner = st.selectbox('Has Partner?', ['Yes', 'No'])
dependents = st.selectbox('Has Dependents?', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service?', ['Yes', 'No'])
paperless_billing = st.selectbox('Paperless Billing?', ['Yes', 'No'])

internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No'])

# --- Preprocessing Function ---
def preprocess_input(tenure, monthly_charges, total_charges, contract_type, payment_method,
                     gender, partner, dependents, phone_service, paperless_billing,
                     internet_service, online_security, tech_support):
    binary_map = lambda x: 1 if x == 'Yes' else 0

    contract_map = {'One year': [1, 0], 'Two year': [0, 1], 'Month-to-month': [0, 0]}
    payment_map = {
        'Bank transfer': [1, 0, 0],
        'Credit card': [0, 1, 0],
        'Mailed check': [0, 0, 1],
        'Electronic check': [0, 0, 0]
    }
    internet_map = {'DSL': [1, 0], 'Fiber optic': [0, 1], 'No': [0, 0]}
    online_security_map = {'No': [1], 'Yes': [0]}
    tech_support_map = {'No': [1], 'Yes': [0]}

    # Manually scaled inputs (min and max values from dataset)
    tenure_scaled = tenure / 72
    monthly_charges_scaled = monthly_charges / 120  # assume ~120 max
    total_charges_scaled = total_charges / 9000     # assume ~9000 max

    features = [
        1 if gender == 'Male' else 0,
        binary_map(partner),
        binary_map(dependents),
        binary_map(phone_service),
        tenure_scaled,
        monthly_charges_scaled,
        total_charges_scaled,
        binary_map(paperless_billing)
    ]
    features += contract_map[contract_type]
    features += payment_map[payment_method]
    features += internet_map[internet_service]
    features += online_security_map[online_security]
    features += tech_support_map[tech_support]

    padded = np.pad(features, (0, 35 - len(features)), mode='constant')
    return np.array([padded])

# --- Predict Button ---
if st.button("🔍 Predict Churn"):
    input_data = preprocess_input(
        tenure, monthly_charges, total_charges,
        contract_type, payment_method,
        gender, partner, dependents, phone_service, paperless_billing,
        internet_service, online_security, tech_support
    )

    prediction = model.predict(input_data)[0][0]

    st.markdown("### 🔎 Result:")
    if prediction > 0.5:
        st.error(f"⚠️ High likelihood of churn: **{prediction:.2f}**")
    else:
        st.success(f"✅ Low likelihood of churn: **{prediction:.2f}**")
