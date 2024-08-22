# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:05:37 2024

@author: affu4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = r"C:\Users\affu4\Downloads\drive-download-20240806T044506Z-001\Housing.csv"
data = pd.read_csv(file_path)

# Check the dataset structure
print(data.info())

# Handle missing values for numerical columns with the median
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for column in numerical_columns:
    data[column].fillna(data[column].median(), inplace=True)

# Handle missing values for categorical columns with the mode
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Separate features and target variable
X = data.drop(columns=['price'])  
y = data['price']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing and training pipeline for Linear Regression
linear_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Create preprocessing and training pipeline for Random Forest
random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model_pipeline.fit(X_train, y_train)

# Make predictions with Linear Regression
y_pred_linear = linear_model_pipeline.predict(X_test)

# Evaluate the Linear Regression model
rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f'Linear Regression RMSE: {rmse_linear}')
print(f'Linear Regression MAE: {mae_linear}')
print(f'Linear Regression R²: {r2_linear}')

# Train the Random Forest model
random_forest_pipeline.fit(X_train, y_train)

# Make predictions with Random Forest
y_pred_rf = random_forest_pipeline.predict(X_test)

# Evaluate the Random Forest model
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest RMSE: {rmse_rf}')
print(f'Random Forest MAE: {mae_rf}')
print(f'Random Forest R²: {r2_rf}')
