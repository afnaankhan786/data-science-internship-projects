# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:15:37 2024

@author: affu4
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load your dataset (assuming a CSV file with a 'review' column)
file_path = r"C:\Users\affu4\Downloads\drive-download-20240808T053916Z-001\apple_iphone_11_reviews.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, float):
        return ''
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(tokens)

data['cleaned_review'] = data['review_title'].apply(preprocess_text)

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

data['sentiment'] = data['cleaned_review'].apply(get_sentiment)

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Distribution of Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()