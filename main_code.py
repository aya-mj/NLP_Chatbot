import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer 
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import re
import time
from collections import Counter
import numpy as np

# Uncomment to download necessary resources
# # Download necessary resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('book')
# nltk.download('words')

# Set up stop words
stop_words = set(stopwords.words('english'))
dictionary = set(words.words())

# Create a frequency dictionary from available texts for better spell checking
def create_frequency_dict():
    available_texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
    all_words = []
    for text in available_texts:
        all_words.extend([word.lower() for word in text if word.isalpha()])
    return Counter(all_words)

# Create frequency dictionary
word_freq_dict = create_frequency_dict()

# Enhanced text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Fixed invalid escape sequence by using r prefix
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

def get_word_frequency(word, freq_dict):
    """Get frequency of a word in the frequency dictionary"""
    return freq_dict.get(word, 0)

def get_suggestions(word, word_set, max_distance=2):
    """Get multiple suggestions for a misspelled word"""
    suggestions = []
    for dict_word in word_set:
        dist = edit_distance(word, dict_word)
        if dist <= max_distance:
            suggestions.append((dict_word, dist))
    # Limit number of suggestions to improve performance
    return sorted(suggestions, key=lambda x: x[1])[:10]

def score_suggestion(suggestion, context_words, freq_dict):
    """Score a suggestion based on context and frequency"""
    word, distance = suggestion
    # Base score is inverse of edit distance
    score = 1 / (distance + 1)
    
    # Add frequency bonus
    freq = get_word_frequency(word, freq_dict)
    score += freq * 0.1
    
    # Add context bonus if word appears in context
    if word in context_words:
        score += 0.5
    
    return score

def spell_check(misspelled_word, word_set, freq_dict, context_words=None):
    """Enhanced spell check with context awareness"""
    if context_words is None:
        context_words = []
    
    # Return the original word if it's in the dictionary
    if misspelled_word in word_set:
        return misspelled_word
    
    # Get all possible suggestions
    suggestions = get_suggestions(misspelled_word, word_set)
    
    if not suggestions:
        return misspelled_word  # Return original if no suggestions found
    
    # Score and sort suggestions
    scored_suggestions = [(suggestion, score_suggestion(suggestion, context_words, freq_dict)) 
                         for suggestion in suggestions]
    scored_suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best suggestion
    return scored_suggestions[0][0][0]

def correct_spelling(text, word_set, freq_dict):
    """Enhanced spelling correction with context awareness"""
    words_in_text = word_tokenize(text)
    corrected_words = []
    
    # Create context window
    context_window = 3
    for i, word in enumerate(words_in_text):
        if not word.isalpha():
            corrected_words.append(word)
            continue
            
        # Get context words
        start = max(0, i - context_window)
        end = min(len(words_in_text), i + context_window + 1)
        context_words = [w.lower() for w in words_in_text[start:end] if w.isalpha()]
        
        if word.lower() in word_set:
            corrected_words.append(word)
        else:
            # Get corrected word with context
            corrected_word = spell_check(word.lower(), word_set, freq_dict, context_words)
            # Preserve original case
            if word[0].isupper():
                corrected_word = corrected_word.capitalize()
            corrected_words.append(corrected_word)
    
    return " ".join(corrected_words)

# Function to prepare training data from NLTK book texts
def prepare_training_data():
    # List of available texts in nltk.book
    available_texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
    text_names = ['Moby Dick', 'Sense and Sensibility', 'The Book of Genesis', 
                 'Inaugural Addresses', 'Chat Corpus', 'Monty Python', 
                 'Wall Street Journal', 'Personals Corpus', 'The Man Who Was Thursday']
    
    all_processed_sentences = []
    text_sources = []
    
    for idx, text_obj in enumerate(available_texts):
        print(f"Processing {text_names[idx]}...")
        
        # Convert text object to string
        text_str = ' '.join(text_obj)
        
        # Split into sentences
        sentences = sent_tokenize(text_str)
        
        # Process each sentence
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Only include sentences with at least 3 words
                processed = process_text(sentence)[0]
                all_processed_sentences.append(processed)
                text_sources.append(text_names[idx])
    
    # Create DataFrame with processed texts
    df = pd.DataFrame({
        'processed_text': all_processed_sentences,
        'source': text_sources
    })
    
    print(f"Total sentences processed: {len(df)}")
    return df

# Example usage
if __name__ == "__main__":
    # Apply spell check to the text
    input_sentence = "Thiss is a smple sentnce with soome mistake"
    corrected = correct_spelling(input_sentence, dictionary, word_freq_dict)

    print("Corrected Sentence:")
    print(corrected)