# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:50:17 2024

@author: affu4
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# File path
file_path = r"C:\Users\affu4\Downloads\drive-download-20240801T071213Z-001\bank_transactions.csv"

# Load data
df = pd.read_csv(file_path, usecols=['TransactionTime', 'TransactionAmount (INR)'])
print(df.head())
print(df.dtypes)
# Preprocessing
non_numeric_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[non_numeric_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(non_numeric_cols))

df_numeric = df.drop(columns=non_numeric_cols)
df_encoded = pd.concat([df_numeric, encoded_df], axis=1)

imputer = SimpleImputer(strategy='mean')
df_cleaned = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

# Feature Selection
selector = SelectKBest(score_func=f_classif, k='all')
X_reduced = selector.fit_transform(df_cleaned, df_cleaned['TransactionTime'])  # Replace 'Amount' with target if applicable

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Incremental PCA
ipca = IncrementalPCA(n_components=1)  # Adjust components based on needs
chunk_size = 100000  # Adjust based on system memory

for chunk in pd.read_csv(file_path, usecols=df_encoded.columns, chunksize=chunk_size):
    chunk_encoded = encoder.transform(chunk[non_numeric_cols])
    chunk_cleaned = pd.concat([chunk.drop(columns=non_numeric_cols), pd.DataFrame(chunk_encoded, columns=encoder.get_feature_names_out(non_numeric_cols))], axis=1)
    chunk_cleaned = pd.DataFrame(imputer.transform(chunk_cleaned), columns=df_encoded.columns)
    chunk_scaled = scaler.transform(chunk_cleaned)
    ipca.partial_fit(chunk_scaled)

X_pca = ipca.transform(X_scaled)

# Clustering
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
df_cleaned['KMeans_Cluster'] = kmeans.fit_predict(X_pca)

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = MiniBatchKMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K-means Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
