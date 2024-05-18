#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:55:40 2024

@author: aviralh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score

# Load the dataset
file_path = './filtered_eda_columns.csv'  # Replace with the correct path to your CSV file
df = pd.read_csv(file_path)

# Split the data into features and target variable
X = df.drop('Fraud', axis=1)
y = df['Fraud']

# Split the data into training (80%) and testing (20%) sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'AUC-ROC: {auc_roc}')
print(f'F1 Score: {f1}')
print(f'Recall: {recall}')
