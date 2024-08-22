# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:37:15 2024

@author: affu4
"""

import pandas as pd

# Load the dataset
file_path = r"C:\Users\affu4\Downloads\drive-download-20240813T055929Z-001\Anime_data.csv"
anime_data = pd.read_csv(file_path)

# Display the first few rows and summary information to understand the dataset
anime_data.head(), anime_data.info(), anime_data.describe()

anime_data = anime_data[pd.to_numeric(anime_data['Title'], errors='coerce').notna()]

# Drop rows with missing values
anime_data_cleaned = anime_data.dropna()

# Verify the changes by checking the shape of the cleaned dataset and display the first few rows
anime_data_cleaned.shape, anime_data_cleaned.head()

print(anime_data_cleaned[['ScoredBy', 'Anime_id', 'Rating']].head())
print(anime_data_cleaned[['ScoredBy', 'Anime_id', 'Rating']].shape)
print(anime_data_cleaned.shape)

user_item_matrix = anime_data_aggregated.pivot(index='ScoredBy', columns='Anime_id', values='Rating').fillna(0)
print(user_item_matrix.head())
print(user_item_matrix.shape)

print(anime_data_cleaned.dtypes)
# Sample subset for testing
sample_data = anime_data_cleaned.sample(10)
sample_matrix = sample_data.pivot(index='ScoredBy', columns='Anime_id', values='Rating').fillna(0)
print(sample_matrix)
print(sample_matrix.shape)


from sklearn.preprocessing import MinMaxScaler

# Selecting numerical columns for normalization
numerical_cols = ['Rating', 'ScoredBy', 'Popularity', 'Members', 'Episodes']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected numerical columns
anime_data_cleaned[numerical_cols] = scaler.fit_transform(anime_data_cleaned[numerical_cols])

# Display the first few rows of the normalized dataset
anime_data_cleaned.head()

# Check for duplicates in the 'ScoredBy' column
duplicates = anime_data_cleaned['ScoredBy'].duplicated().sum()
print(f"Number of duplicates in 'ScoredBy': {duplicates}")



# Group by 'ScoredBy' and 'Anime_id' and calculate the mean rating
anime_data_aggregated = anime_data_cleaned.groupby(['ScoredBy', 'Anime_id']).mean().reset_index()

# Create the user-item interaction matrix
user_item_matrix = anime_data_aggregated.pivot(index='ScoredBy', columns='Anime_id', values='Rating').fillna(0)

# Check the shape of the user-item matrix
print(f"User-item matrix shape: {user_item_matrix.shape}")

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Convert to a sparse matrix
user_item_sparse = csr_matrix(user_item_matrix)

# Check the shape of the sparse matrix
print(f"Sparse matrix shape: {user_item_sparse.shape}")

# Apply SVD
if user_item_sparse.shape[0] > 0 and user_item_sparse.shape[1] > 0:
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_svd = svd.fit_transform(user_item_sparse)
    print(f"SVD matrix shape: {matrix_svd.shape}")
    print(matrix_svd[:5])
else:
    raise ValueError("The sparse matrix is empty. Check data preprocessing steps.")

