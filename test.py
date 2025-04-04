import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import time
import matplotlib.pyplot as plt
from collections import Counter

# Download necessary resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('book')

# Set up stop words
stop_words = set(stopwords.words('english'))

# Enhanced text cleaning function
def clean_text(text):
    """Clean text by removing non-alphabetic characters and stopwords"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()  # Split text into words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Function to measure performance
def measure_performance(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Fixed process_text function to handle string input properly
@measure_performance
def process_text(text):
    """Process a single text using the NLP pipeline"""
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('cleaner', FunctionTransformer(clean_text, validate=False)),  # Disable validation for text processing
        ('tokenizer', FunctionTransformer(lambda x: word_tokenize(x), validate=False)),
        ('stemmer', FunctionTransformer(lambda x: [PorterStemmer().stem(word) for word in x], validate=False)),
        ('joiner', FunctionTransformer(lambda x: ' '.join(x), validate=False))
    ])
    
    processed = preprocessor.transform(text)
    return processed

# Process multiple texts efficiently
@measure_performance
def process_multiple_texts(texts):
    """Process a list of texts using the NLP pipeline"""
    return [process_text(text) for text in texts]

# Analyze word frequency
def analyze_word_frequency(processed_texts, top_n=20):
    """Analyze word frequency in processed texts"""
    all_words = ' '.join(processed_texts).split()
    word_counts = Counter(all_words)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    
    # Create DataFrame for visualization
    freq_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    return freq_df

# Visualize word frequency
def plot_word_frequency(freq_df):
    """Plot word frequency as a bar chart"""
    plt.figure(figsize=(12, 6))
    plt.bar(freq_df['Word'], freq_df['Count'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top Word Frequencies')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    return plt

# Main analysis function
@measure_performance
def analyze_text(sample_size=5000):
    """Run the complete text analysis pipeline"""
    # Sample the text
    text1_sample = ' '.join(text1[:sample_size])
    
    # Create multiple documents by splitting the text
    sentences = sent_tokenize(text1_sample)
    processed_texts = [clean_text(sentence) for sentence in sentences]
    
    # Fit the vectorizer on multiple documents
    vectorizer = TfidfVectorizer(min_df=2)  # Ignore terms that appear in less than 2 documents
    X = vectorizer.fit_transform(processed_texts)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create a DataFrame for visualization
    tfidf_df = pd.DataFrame(
        X.toarray(),
        columns=feature_names
    )
    
    # Find the most important terms for each document
    top_terms = []
    for i, doc in enumerate(sentences):
        if len(doc.split()) < 3:  # Skip very short sentences
            continue
        tfidf_row = tfidf_df.iloc[i]
        sorted_indices = np.argsort(tfidf_row.values)[::-1]
        top_indices = sorted_indices[:5]  # Get top 5 terms
        top_terms.append({
            'document': doc,
            'top_terms': [(feature_names[idx], tfidf_row[feature_names[idx]]) for idx in top_indices]
        })
    
    # Get word frequency analysis
    freq_df = analyze_word_frequency(processed_texts)
    
    return {
        'tfidf_matrix': X,
        'tfidf_df': tfidf_df,
        'feature_names': feature_names,
        'top_terms': top_terms,
        'word_freq': freq_df,
        'processed_texts': processed_texts
    }

# Execute the analysis (uncomment to run)
results = analyze_text()

print("\nMost frequent words:")
print(results['word_freq'].head(10))