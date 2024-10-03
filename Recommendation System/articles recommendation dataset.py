# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:44:54 2024

@author: affu4
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

### Step 1: Data Preprocessing (Normalization and Handling Missing Values)

# Load the dataset
file_path = r"C:\Users\affu4\Downloads\drive-download-20240813T055929Z-001\articles recommendation dataset.csv"
dataset = pd.read_csv(file_path, encoding='ISO-8859-1')

# Handle missing values (if any)
dataset.dropna(inplace=True)

# Function to preprocess text (normalization)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    return text

# Apply preprocessing to the 'Article' column
dataset['Processed_Article'] = dataset['Article'].apply(preprocess_text)

### Step 2: Content-Based Filtering

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the 'Processed_Article' column to create TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['Processed_Article'])

# Calculate the cosine similarity between the TF-IDF vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert the cosine similarity matrix to a DataFrame for better readability
cosine_sim_df = pd.DataFrame(cosine_sim, index=dataset['Title'], columns=dataset['Title'])

# Function to recommend articles based on content similarity
def recommend_articles_content(title, cosine_sim_df, top_n=5):
    sim_scores = cosine_sim_df[title].sort_values(ascending=False)
    return sim_scores.iloc[1:top_n+1]

# Example: Recommend top 5 articles similar to a given article
recommended_articles = recommend_articles_content('Assumptions of Machine Learning Algorithms', cosine_sim_df, top_n=5)
print("Content-Based Filtering Recommendations:")
print(recommended_articles)

