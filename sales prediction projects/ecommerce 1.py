# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:40:11 2024

@author: affu4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the new dataset
ecommerce_data_path = r"C:\Users\affu4\Downloads\drive-download-20240806T044506Z-001\ecommerce_product_dataset.csv"
ecommerce_data = pd.read_csv(ecommerce_data_path)

# Separate numerical and categorical columns
numerical_cols = ecommerce_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = ecommerce_data.select_dtypes(include=['object']).columns

# Handle missing values for numerical columns
num_imputer = SimpleImputer(strategy='mean')
ecommerce_data[numerical_cols] = num_imputer.fit_transform(ecommerce_data[numerical_cols])

# Handle missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
ecommerce_data[categorical_cols] = cat_imputer.fit_transform(ecommerce_data[categorical_cols])

# Encode categorical variables using one-hot encoding
ecommerce_data_encoded = pd.get_dummies(ecommerce_data, columns=categorical_cols)

# Assuming 'target_column' is the column to predict (replace with the actual column name)
target_column = 'Sales'
X = ecommerce_data_encoded.drop(target_column, axis=1)
y = ecommerce_data_encoded[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

# Train a random forest regression model
random_forest_regressor = RandomForestRegressor(random_state=42)
random_forest_regressor.fit(X_train, y_train)
y_pred_rf = random_forest_regressor.predict(X_test)

# Train a HistGradientBoostingRegressor model
hist_gradient_boosting_regressor = HistGradientBoostingRegressor(random_state=42)
hist_gradient_boosting_regressor.fit(X_train, y_train)
y_pred_hgb = hist_gradient_boosting_regressor.predict(X_test)

# Evaluate model performance
linear_regression_performance = {
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred_linear),
    "Mean Squared Error": mean_squared_error(y_test, y_pred_linear),
    "R-squared": r2_score(y_test, y_pred_linear)
}

random_forest_performance = {
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred_rf),
    "Mean Squared Error": mean_squared_error(y_test, y_pred_rf),
    "R-squared": r2_score(y_test, y_pred_rf)
}

hist_gradient_boosting_performance = {
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred_hgb),
    "Mean Squared Error": mean_squared_error(y_test, y_pred_hgb),
    "R-squared": r2_score(y_test, y_pred_hgb)
}

# Prepare a new data sample for prediction (replace with actual new data)
new_data = pd.DataFrame({
    # Add appropriate columns and values for the new data sample
    'ProductID': [1],
    'ProductName': ['Headphones'],
    'Category': ['Electronics'],
    'Price': [400.31],
    'Rating': [1.7],
    'NumReviews': [3772],
    'StockQuantity': [20],
    'Discount': [0.08],
    'Sales': [466],
    'DateAdded': [2023-11-10],
    'City': ['Albuquerque'],
})

# Preprocess the new data sample
new_data[categorical_cols] = cat_imputer.transform(new_data[categorical_cols])
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols)
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions using the best model (choose the model with the highest R-squared)
best_model = hist_gradient_boosting_regressor  # Assuming HistGradientBoostingRegressor performed best
predictions = best_model.predict(new_data_encoded)

linear_regression_performance, random_forest_performance, hist_gradient_boosting_performance, predictions
