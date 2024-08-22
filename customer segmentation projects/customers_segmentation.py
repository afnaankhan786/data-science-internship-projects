# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:42:09 2024

@author: affu4
"""

import pandas as pd

# Load the data
file_path = 'C:\\Users\\affu4\\Downloads\\drive-download-20240801T071213Z-001\\customer_segmentation_data.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Display basic information about the dataframe
print(df.info())
# Display the data types of each column
print(df.dtypes)
from sklearn.preprocessing import LabelEncoder

# Example: Convert a categorical column to numeric
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])  # Replace 'Gender' with your actual column name

# Example: Convert categorical columns to dummy/indicator variables
df_encoded = pd.get_dummies(df, columns=['gender', 'preferred_category'])  # Replace with your actual column names

# If you had columns that need to be dropped
df_encoded = df_encoded.drop([ 'age', 'income'], axis=1)  # Drop columns that are not needed
# Standardize features
from sklearn.preprocessing import StandardScaler

features = df_encoded[['last_purchase_amount', 'purchase_frequency']]  # Include all numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

# Apply Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hc_clusters = hc.fit_predict(scaled_features)
df['HC_Cluster'] = hc_clusters

# Visualize K-Means Clusters
import matplotlib.pyplot as plt

plt.scatter(df['last_purchase_amount'], df['purchase_frequency'], c=df['Cluster'], cmap='viridis')
plt.xlabel('last_purchase_amount')
plt.ylabel('purchase_frequency')
plt.title('K-Means Clustering')
plt.show()

# Visualize Hierarchical Clustering
plt.scatter(df['last_purchase_amount'], df['purchase_frequency'], c=df['HC_Cluster'], cmap='viridis')
plt.xlabel('last_purchase_amount')
plt.ylabel('purchase_frequency')
plt.title('Hierarchical Clustering')
plt.show()


