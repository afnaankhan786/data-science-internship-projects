# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:07:43 2024

@author: affu4
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the email spam dataset
file_path_spam = r"C:\Users\affu4\Downloads\drive-download-20240808T060116Z-001\email_spam.csv"
df_spam = pd.read_csv(file_path_spam)

# Display the first few rows and basic information about the dataset
print("First few rows of email_spam.csv:")
print(df_spam.head())

print("\nSummary of email_spam.csv:")
print(df_spam.info())

print("\nMissing values in email_spam.csv:")
print(df_spam.isnull().sum())

print("\nDescriptive statistics of email_spam.csv:")
print(df_spam.describe(include='all'))

# Encode the target column 'type' as binary (0 for not spam, 1 for spam)
label_encoder = LabelEncoder()
df_spam['type'] = label_encoder.fit_transform(df_spam['type'])

# Convert text data to numerical data using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df_spam['text'])

# Target variable
y = df_spam['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train and evaluate Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_smote, y_train_smote)
y_pred_log_reg = log_reg.predict(X_test)

print("Logistic Regression Performance:")
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

# Train and evaluate Decision Tree model
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_smote, y_train_smote)
y_pred_tree = tree_clf.predict(X_test)

print("Decision Tree Performance:")
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# ROC-AUC for Logistic Regression
log_reg_roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1])
fpr, tpr, thresholds = roc_curve(y_test, log_reg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % log_reg_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import joblib

# Save the trained Logistic Regression model
joblib.dump(log_reg, 'logistic_regression_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Load the saved model and vectorizer
log_reg = joblib.load('logistic_regression_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def preprocess_and_predict(text):
    # Transform the text using the TF-IDF vectorizer
    text_vectorized = tfidf.transform([text])
    
    # Predict using the logistic regression model
    prediction = log_reg.predict(text_vectorized)
    
    return prediction[0]

# Example usage
new_email = "Congratulations! You've won a $1000 gift card. Click here to claim your prize."
prediction = preprocess_and_predict(new_email)
print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
