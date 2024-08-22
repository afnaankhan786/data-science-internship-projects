# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:02:47 2024

@author: affu4
"""

import pandas as pd

# Load the data
data = pd.read_csv(r"C:\Users\affu4\Downloads\drive-download-20240801T071213Z-001\Mall_Customers.csv")


# Use a smaller subset for initial exploration
data_subset = data.sample(frac=0.1, random_state=42)  # Adjust the fraction as needed
data_subset.head()

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variables
label_encoders = {}
for column in ['Gender']:
    le = LabelEncoder()
    data_subset[column] = le.fit_transform(data_subset[column])
    label_encoders[column] = le

# Drop irrelevant columns
data_processed = data_subset.drop(columns=['Gender'])

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_processed)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# Apply hierarchical clustering on a smaller subset
subset_for_hc = data_scaled[:1000]  # Using the first 1000 rows

# Perform linkage on the subset
linked = linkage(subset_for_hc, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Apply K-means with the optimal number of clusters
optimal_clusters = 5  # Adjust based on elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data_subset['Cluster'] = kmeans.fit_predict(data_scaled)

# Analyze the clusters
cluster_analysis = data_subset.groupby('Cluster').mean()
cluster_analysis

import seaborn as sns

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='Purchase', hue='Cluster', data=data_subset, palette='viridis')
plt.title('Clusters Visualization')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()



# Use a smaller subset for initial exploration
data_subset = data.sample(frac=0.1, random_state=42)  # Adjust the fraction as needed
data_subset.head()

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']:
    le = LabelEncoder()
    data_subset[column] = le.fit_transform(data_subset[column])
    label_encoders[column] = le

# Drop irrelevant columns
data_processed = data_subset.drop(columns=['User_ID', 'Product_ID'])

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_processed)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# Apply hierarchical clustering on a smaller subset
subset_for_hc = data_scaled[:1000]  # Using the first 1000 rows

# Perform linkage on the subset
linked = linkage(subset_for_hc, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Apply K-means with the optimal number of clusters
optimal_clusters = 5  # Adjust based on elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data_subset['Cluster'] = kmeans.fit_predict(data_scaled)

# Analyze the clusters
cluster_analysis = data_subset.groupby('Cluster').mean()
cluster_analysis

import seaborn as sns

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Spending Score (1-100)', y='Gender', hue='Cluster', data=data_subset, palette='viridis')
plt.title('Clusters Visualization')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Gender')
plt.legend(title='Cluster')
plt.show()

