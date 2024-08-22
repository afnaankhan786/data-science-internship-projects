# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:22:58 2024

@author: affu4
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv(r"C:\Users\affu4\Downloads\drive-download-20240808T060116Z-001\CreditCardData.csv")
print(data)
# Convert date feature to numeric
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Date'] = data['Date'].astype('int64')  # Convert to Unix timestamp

# Define features and their types
categorical_features = ['Transaction ID', 'Amount']  # Replace with actual categorical columns
numeric_features = ['Time', 'Fraud']  # Replace with actual numeric columns

# Define preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # Use sparse_output instead of sparse
        ]), categorical_features)
    ],
    remainder='passthrough'
)

# Define the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', MaxAbsScaler())  # Use MaxAbsScaler to handle sparse data
])

# Define features and target
X = data[numeric_features + categorical_features + ['Date']]
y = data['Fraud']  # Replace with the actual target column name

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply preprocessing
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

# Handle Imbalanced Data
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Train Model
model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

# Evaluate Model
y_pred = model.predict(X_test_processed)
print(classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))

# Save Model and Pipeline
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.joblib')
joblib.dump(model, 'model.joblib')

# Initialize the Decision Tree model
decision_tree_model = DecisionTreeClassifier()

# Train the model on the resampled data
decision_tree_model.fit(X_train_resampled, y_train_resampled)


# Define parameter distributions
param_distributions = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(),
    param_distributions=param_distributions,
    n_iter=10,  # Number of parameter settings to sample
    cv=5,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_train_resampled, y_train_resampled)

# Print best parameters
print("Best Parameters:", random_search.best_params_)

# Use the best model from random search
best_tree_model = random_search.best_estimator_

# Predict and evaluate
y_pred_best_tree = best_tree_model.predict(X_test_processed)
print("Best Decision Tree Performance:")
print(classification_report(y_test, y_pred_best_tree))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred_best_tree))

# Load the preprocessing pipeline and decision tree model
preprocessing_pipeline = joblib.load('preprocessing_pipeline.joblib')
decision_tree_model = joblib.load('decision_tree_model.joblib')

# Check if the objects are loaded
if preprocessing_pipeline is None or decision_tree_model is None:
    raise ValueError("Failed to load preprocessing pipeline or decision tree model.")

# Function to preprocess and predict new data
def predict_fraud(new_data):
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("Input new_data must be a pandas DataFrame")
    
    # Apply preprocessing
    if preprocessing_pipeline:
        new_data_processed = preprocessing_pipeline.transform(new_data)
    else:
        raise ValueError("Preprocessing pipeline is not available.")
    
    # Make predictions
    predictions = decision_tree_model.predict(new_data_processed)
    
    return predictions


# Example timestamp
timestamp = pd.Timestamp('2024-08-08')

# Convert to Unix timestamp (seconds since epoch)
timestamp_int = int(timestamp.timestamp())
print("Unix Timestamp:", timestamp_int)

# Example timestamp
timestamp = pd.Timestamp('2024-08-08')

# Extract components
year = timestamp.year
month = timestamp.month
day = timestamp.day

print(f"Year: {year}, Month: {month}, Day: {day}")

# Example data
data = pd.DataFrame({
    'Fraud': [1.5, 3.2, 4.0],
    'Transaction ID': ['A', 'B', 'A'],
    'Date': ['2024-08-08', '2024-08-09', '2024-08-10']
})
print("Columns in data:", data.columns)

# Convert date feature
data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: int(x.timestamp()))

# Define preprocessing
numeric_features = ['Fraud']
categorical_features = ['Transaction ID']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)  # Updated parameter
    ]
)

# Define the model
model = RandomForestClassifier()

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Fit the pipeline
X_train = data
y_train = [0, 1, 0]  # Example target variable
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_train)
print("Predictions:", predictions)