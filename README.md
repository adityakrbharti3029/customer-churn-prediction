# 📊 Customer Churn Prediction App

A web application built with Streamlit and Scikit-learn to predict customer churn based on their data. This project demonstrates an end-to-end machine learning workflow, from data cleaning and model training to deployment as an interactive web app.

## 🚀 Overview

This app allows users to input various customer attributes such as tenure, contract type, and monthly charges. Based on this input, a pre-trained logistic regression model predicts whether the customer is likely to churn or stay loyal, along with the probability for the prediction.

## ✨ Features

- **Interactive UI:** A clean and user-friendly interface built with Streamlit.
- **Real-time Predictions:** Get instant churn predictions as you change the input parameters.
- **Probability Score:** Shows the confidence of the model in its prediction.
- **Data Visualization:** Includes a Plotly chart to visualize the loyalty vs. churn probability.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Streamlit, Pandas, Scikit-learn, Plotly
- **Model:** Logistic Regression

## 📂 Project Structure

- `app.py`: Contains the code for the Streamlit web application.
- `model.py`: The script used to train the logistic regression model and save it as a `.pkl` file.
- `churn_model_pipeline.pkl`: The saved, pre-trained machine learning model.
- `requirements.txt`: Lists all the necessary Python libraries for the project.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for training the model.
