# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:19:10 2024

@author: affu4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
car_sales_data = pd.read_csv(r"C:\Users\affu4\Downloads\drive-download-20240806T044506Z-001\Car_sales.csv")

numerical_cols = car_sales_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = car_sales_data.select_dtypes(include=['object']).columns

# Handle missing values
num_imputer = SimpleImputer(strategy='mean')
car_sales_data[numerical_cols] = num_imputer.fit_transform(car_sales_data[numerical_cols])

# Encode categorical variables using one-hot encoding
# Handle missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
car_sales_data[categorical_cols] = cat_imputer.fit_transform(car_sales_data[categorical_cols])

# Encode categorical variables using one-hot encoding
car_sales_data_encoded = pd.get_dummies(car_sales_data, columns=categorical_cols)

# Split the data into features and target variable
X = car_sales_data_encoded.drop('Sales_in_thousands', axis=1)
y = car_sales_data_encoded['Sales_in_thousands']

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
print("Linear Regression Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_linear))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("R-squared:", r2_score(y_test, y_pred_linear))

print("\nRandom Forest Regression Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R-squared:", r2_score(y_test, y_pred_rf))

print("\nHistGradientBoostingRegressor Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_hgb))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_hgb))
print("R-squared:", r2_score(y_test, y_pred_hgb))

# Make predictions using the best model (choose based on R-squared)
best_model = hist_gradient_boosting_regressor
new_data = pd.DataFrame({
    'Manufacturer': ['Ford'],
    'Model': ['Fusion'],
    'Vehicle_type': ['Passenger'],
    'Latest_Launch': ['1/1/2021'],
    'Price_in_thousands': [25.0],
    'Engine_size': [2.0],
    'Horsepower': [175],
    'Wheelbase': [105],
    'Width': [70],
    'Length': [190],
    'Curb_weight': [3200],
    'Fuel_capacity': [16],
    'Fuel_efficiency': [30],
    'Power_perf_factor': [65]
})

# Preprocess the new data
new_data[categorical_cols] = cat_imputer.transform(new_data[categorical_cols])
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols)
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions
predictions = best_model.predict(new_data_encoded)
print("\nPredictions for new data:")
print(predictions)

