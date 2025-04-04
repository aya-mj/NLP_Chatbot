import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
import re
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import time
from nltk.metrics.distance import edit_distance

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('book')
nltk.download('words')

# Set up stop words
stop_words = set(stopwords.words('english'))
dictionary = set(words.words())

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

# Spell check function
def spell_check(misspelled_word, dictionary):
    closest_word = min(dictionary, key=lambda w: edit_distance(misspelled_word, w))
    return closest_word

def correct_spelling(text, dictionary):
    
    words_in_text = word_tokenize(text)
    corrected_words = []
    for word in words_in_text:
        # If word is in dictionary, assume it's correct; so it remains unchanged
        if word.lower() in dictionary:
            corrected_words.append(word)
        else:
            # If word is not in dictionary, check if it can be corrected
            suggestions = spell_check(word.lower(), dictionary)
            corrected_words.append(suggestions)
    return " ".join(corrected_words)


# Example usage
# text1_sample = ' '.join(text1[:5000])  # Sample the first 5000 words


# # Create multiple documents by splitting the text
# sentences = sent_tokenize(text1_sample)
# processed_texts = [clean_text(sentence) for sentence in sentences]

# # Now fit the vectorizer on multiple documents
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(processed_texts)

# # Get feature names
# feature_names = vectorizer.get_feature_names_out()

# # Create a DataFrame for better visualization
# tfidf_df = pd.DataFrame(
#     X.toarray(),
#     columns=feature_names
# )

# Apply spell check to the text
input_sentence = "Thiss is a smple sentnce with soome mistake"
corrected = correct_spelling(input_sentence, dictionary)

print("Corrected Sentence:")
print(corrected)