import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

print("Script execution started...")

# Step 1: Load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. Please check the file path.")
    exit()

# Step 2: Clean and preprocess the data
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("Data cleaning complete.")
print(df.head())

# Step 3: Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 4: Define preprocessing steps for numeric and categorical features
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 5: Define the model
model = LogisticRegression(max_iter=1000)

# Step 6: Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Step 7: Train the pipeline
print("Model training started...")
pipeline.fit(X, y)
print("Model training complete.")

# Step 8: Save the trained pipeline to a file
joblib.dump(pipeline, 'churn_model_pipeline.pkl')
print("Model pipeline has been saved as 'churn_model_pipeline.pkl'.")