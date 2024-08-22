# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:37:23 2024

@author: affu4
"""

import pandas as pd

# Load the dataset
file_path = r"C:\Users\affu4\Downloads\drive-download-20240808T060116Z-001\transaction_dataset.csv"
df = pd.read_csv(file_path)

# Display the first few rows and summary information of the dataset
df.head(), df.info(), df.describe()

# Dropping irrelevant columns
df_cleaned = df.drop(columns=['Unnamed: 0', 'Index', 'Address'])

# Checking for missing values
missing_values = df_cleaned.isnull().sum()

# Display columns with missing values and their counts
missing_values[missing_values > 0]

# Dropping columns with missing values
df_cleaned = df_cleaned.dropna(axis=1)

# Verifying that no columns with missing values remain
remaining_missing_values = df_cleaned.isnull().sum()

# Displaying the cleaned dataframe's columns
df_cleaned.columns, remaining_missing_values[remaining_missing_values > 0]

# Check the distribution of the target variable 'FLAG'
class_distribution = df_cleaned['FLAG'].value_counts(normalize=True)
print(class_distribution)

from imblearn.over_sampling import SMOTE

# Separate features and target variable
X = df_cleaned.drop(columns=['FLAG'])
y = df_cleaned['FLAG']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print(y_resampled.value_counts(normalize=True))

from imblearn.under_sampling import RandomUnderSampler

# Apply Random Undersampling to balance the classes
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Check the new class distribution
print(y_resampled.value_counts(normalize=True))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Identify and handle non-numeric columns
print(df.dtypes)

# Drop non-numeric or ID columns if they are not useful
df = df.drop(['Address'], axis=1, errors='ignore')

# Convert categorical columns to numeric
for column in df.select_dtypes(include=['object']).columns:
    if df[column].nunique() < 10:  # Example threshold for label encoding
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
    else:
        df = pd.get_dummies(df, columns=[column])

# Define features and target
X = df.drop('total transactions (including tnx to create contract', axis=1, errors='ignore')
y = df['FLAG_0']

# Drop rows with missing values
X = X.dropna()
y = y[X.index]  # Ensure y is aligned with X

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = decision_tree_model.predict(X_test)
print("Decision Tree Performance:")
print(classification_report(y_test, y_pred_tree))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'], 
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import joblib

joblib.dump(model, 'logistic_model.pkl')

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
logistic_model = joblib.load('logistic_model.pkl')  # Save your model using joblib.dump

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['features']]
    
    # Predict using the model
    prediction = logistic_model.predict(features)
    return jsonify({'fraud': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

import os
print(os.getcwd())

