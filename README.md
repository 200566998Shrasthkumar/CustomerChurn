# 📱 Telco Customer Churn Prediction App

This is a simple web application built using [Streamlit](https://streamlit.io/) that predicts customer churn based on input features. It uses a trained deep learning model (Keras/TensorFlow) to estimate the likelihood of a customer leaving a telecom service provider.

---

## 🚀 Live Demo

👉 [Click here to try the app]([https://your-app-name.streamlit.app](https://customerchurn-2miyr2rfafadndhxo7f7dq.streamlit.app/))  
*(replace with your actual Streamlit Cloud link)*

---

## 📊 Model Overview

The model was trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and includes preprocessing steps such as:

- One-hot encoding of categorical features
- Min-Max scaling of numeric features
- Neural network architecture selected via Keras Tuner

---

## 🔧 Features

- Input form to collect customer information
- Real-time churn prediction
- Friendly and responsive interface
- TensorFlow/Keras model integration
- Deployed on Streamlit Cloud

---

## 🛠 Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy / Pandas
- Git + GitHub for version control

---

## 📝 How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
