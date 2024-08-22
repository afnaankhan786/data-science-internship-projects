# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:53:15 2024

@author: affu4
"""

import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\affu4\Downloads\drive-download-20240808T060116Z-001\online_retail.csv")

# Fill missing descriptions with 'Unknown'
df['Description'].fillna('Unknown', inplace=True)

# Drop rows with missing CustomerID
df_cleaned = df.dropna(subset=['CustomerID'])

# 1. Total Amount Spent
df_cleaned['TotalAmount'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# 2. Extracting Date Features from InvoiceDate
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
df_cleaned['DayOfWeek'] = df_cleaned['InvoiceDate'].dt.dayofweek
df_cleaned['Month'] = df_cleaned['InvoiceDate'].dt.month
df_cleaned['Hour'] = df_cleaned['InvoiceDate'].dt.hour

# 3. Customer Transaction Frequency
df_cleaned['CustomerTransactionCount'] = df_cleaned.groupby('CustomerID')['InvoiceNo'].transform('count')

# 4. Encoding Country using one-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=['Country'], drop_first=True)

# 5. Identifying Transaction Type (Return or Purchase)
df_encoded['TransactionType'] = df_encoded['Quantity'].apply(lambda x: 'Return' if x < 0 else 'Purchase')
df_encoded['TransactionType'] = df_encoded['TransactionType'].map({'Purchase': 0, 'Return': 1})

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Define the target and features
X = df_encoded.drop(columns=['TransactionType', 'InvoiceNo', 'InvoiceDate', 'StockCode', 'Description', 'CustomerID'])
y = df_encoded['TransactionType']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle imbalanced data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_resampled, y_train_resampled)

from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree Model
dec_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dec_tree.fit(X_train_resampled, y_train_resampled)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_dec_tree = dec_tree.predict(X_test)

# Evaluation Metrics for Logistic Regression
log_reg_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_log_reg),
    'Precision': precision_score(y_test, y_pred_log_reg),
    'Recall': recall_score(y_test, y_pred_log_reg),
    'F1 Score': f1_score(y_test, y_pred_log_reg),
    'Confusion Matrix': confusion_matrix(y_test, y_pred_log_reg),
    'ROC AUC': roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
}

# Evaluation Metrics for Decision Tree
dec_tree_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_dec_tree),
    'Precision': precision_score(y_test, y_pred_dec_tree),
    'Recall': recall_score(y_test, y_pred_dec_tree),
    'F1 Score': f1_score(y_test, y_pred_dec_tree),
    'Confusion Matrix': confusion_matrix(y_test, y_pred_dec_tree),
    'ROC AUC': roc_auc_score(y_test, dec_tree.predict_proba(X_test)[:, 1])
}

# Print evaluation metrics
print("Logistic Regression Metrics:", log_reg_metrics)
print("Decision Tree Metrics:", dec_tree_metrics)

# ROC Curve for Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
plt.plot(fpr_log, tpr_log, label="Logistic Regression (area = {:.2f})".format(log_reg_metrics['ROC AUC']))

# ROC Curve for Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, dec_tree.predict_proba(X_test)[:, 1])
plt.plot(fpr_tree, tpr_tree, label="Decision Tree (area = {:.2f})".format(dec_tree_metrics['ROC AUC']))

# Plot ROC Curve
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()

import pickle

# After training your model (using the code provided earlier)

# Save the Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

# Save the Decision Tree model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dec_tree, file)

import pickle

# After training your model (using the code provided earlier)

# Save the Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

# Save the Decision Tree model
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(dec_tree, file)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained models
log_reg_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
dec_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    model_choice = data.get('model', 'logistic_regression')  # Choose model
    
    if model_choice == 'decision_tree':
        prediction = dec_tree_model.predict([data['features']])
    else:
        prediction = log_reg_model.predict([data['features']])
        
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

