# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:59:27 2024

@author: affu4
"""

import pandas as pd

# Load the data
file_path = r'C:\Users\affu4\Downloads\drive-download-20240801T071213Z-001\shopping_behavior_updated.csv'
data = pd.read_csv(file_path)

# Convert categorical variables to numerical values
data_encoded = pd.get_dummies(data, drop_first=True)  # drop_first=True avoids the dummy variable trap

# Inspect the new data
print(data_encoded.head())

from sklearn.preprocessing import LabelEncoder

# Assuming 'Gender' is a categorical column in your dataset
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

label_encoder = LabelEncoder()
data['Item Purchased'] = label_encoder.fit_transform(data['Item Purchased'])

label_encoder = LabelEncoder()
data['Location'] = label_encoder.fit_transform(data['Location'])

label_encoder = LabelEncoder()
data['Size'] = label_encoder.fit_transform(data['Size'])

label_encoder = LabelEncoder()
data['Color'] = label_encoder.fit_transform(data['Color'])

label_encoder = LabelEncoder()
data['Season'] = label_encoder.fit_transform(data['Season'])
label_encoder = LabelEncoder()
data['Frequency of Purchases'] = label_encoder.fit_transform(data['Frequency of Purchases'])
label_encoder = LabelEncoder()
data['Subscription Status'] = label_encoder.fit_transform(data['Subscription Status'])
 
label_encoder = LabelEncoder()
data['Shipping Type'] = label_encoder.fit_transform(data['Shipping Type'])
label_encoder = LabelEncoder()
data['Discount Applied'] = label_encoder.fit_transform(data['Discount Applied'])
label_encoder = LabelEncoder()
data['Promo Code Used'] = label_encoder.fit_transform(data['Promo Code Used'])
label_encoder = LabelEncoder()
data['Previous Purchases'] = label_encoder.fit_transform(data['Previous Purchases'])
label_encoder = LabelEncoder()
data['Payment Method'] = label_encoder.fit_transform(data['Payment Method'])
# Proceed with the rest of your preprocessing
# Check for missing values
print(data.isnull().sum())

# Handle missing values if any
# data = data.fillna(method='ffill')  # Example method

# Normalize or standardize features (if needed)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters (Elbow Method)
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):  # Testing for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with the chosen number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
data_encoded['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
plt.scatter(data_encoded.iloc[:, 0], data_encoded.iloc[:, 1], c=data_encoded['Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

import scipy.cluster.hierarchy as sch

# Create a dendrogram
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

# Apply Hierarchical Clustering
hc = sch.fcluster(sch.linkage(scaled_data, method='ward'), t=optimal_clusters, criterion='maxclust')
data_encoded['HC_Cluster'] = hc

# Visualize Hierarchical Clustering
plt.scatter(data_encoded.iloc[:, 0], data_encoded.iloc[:, 1], c=data_encoded['HC_Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

