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

# Download necessary resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('book')

# Set up stop words
stop_words = set(stopwords.words('english'))

# Enhanced text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()  # Split text into words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Function to measure performance
def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@measure_performance
def process_text(text):
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('cleaner', FunctionTransformer(clean_text)),
        ('tokenizer', FunctionTransformer(lambda x: word_tokenize(x))),
        ('stemmer', FunctionTransformer(lambda x: [PorterStemmer().stem(word) for word in x])),
        ('joiner', FunctionTransformer(lambda x: ' '.join(x)))
    ])
   
    processed = preprocessor.transform(text)
    return [processed]  # Return as a list containing one string
# Example usage
text1_sample = ' '.join(text1[:5000])  # Sample the first 5000 words


# Create multiple documents by splitting the text
sentences = sent_tokenize(text1_sample)
processed_texts = [clean_text(sentence) for sentence in sentences]

# Now fit the vectorizer on multiple documents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for better visualization
tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=feature_names
)