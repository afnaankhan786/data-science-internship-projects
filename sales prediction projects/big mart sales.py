# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:31:47 2024

@author: affu4
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
bigmart_data = pd.read_csv(r"C:\Users\affu4\Downloads\drive-download-20240806T044506Z-001\BigMart Sales Data.csv")

# Separate numerical and categorical columns
numerical_cols = bigmart_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = bigmart_data.select_dtypes(include=['object']).columns

# Handle missing values for numerical columns
num_imputer = SimpleImputer(strategy='mean')
bigmart_data[numerical_cols] = num_imputer.fit_transform(bigmart_data[numerical_cols])

# Handle missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
bigmart_data[categorical_cols] = cat_imputer.fit_transform(bigmart_data[categorical_cols])

# Encode categorical variables using one-hot encoding
bigmart_data_encoded = pd.get_dummies(bigmart_data, columns=categorical_cols)

# Split the data into features and target variable
X = bigmart_data_encoded.drop('Item_Outlet_Sales', axis=1)
y = bigmart_data_encoded['Item_Outlet_Sales']

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

# Provide a sample new data point for prediction
new_data = pd.DataFrame({
    'Item_Identifier': ['FDX07'],
    'Item_Weight': [19.2],
    'Item_Fat_Content': ['Low Fat'],
    'Item_Visibility': [0.016047],
    'Item_Type': ['Snack Foods'],
    'Item_MRP': [249.8092],
    'Outlet_Identifier': ['OUT027'],
    'Outlet_Establishment_Year': [1985],
    'Outlet_Size': ['Medium'],
    'Outlet_Location_Type': ['Tier 3'],
    'Outlet_Type': ['Supermarket Type3']
})

# Preprocess the new data
new_data[categorical_cols] = cat_imputer.transform(new_data[categorical_cols])
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols)
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions
predictions = best_model.predict(new_data_encoded)
print("\nPredictions for new data:")
print(predictions)
