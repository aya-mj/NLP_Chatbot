import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer 
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import re
import time
from collections import Counter
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Uncomment to download necessary resources
# # Download necessary resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('book')
# nltk.download('words')

# Set up stop words
stop_words = set(stopwords.words('english'))
dictionary = set(words.words())

############################__PHASE 1__#############################
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
    tokens = text.split()  
    tokens = [word for word in tokens if word not in stop_words]  
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
    return [processed] 

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
    
    context_window = 3
    for i, word in enumerate(words_in_text):
        if not word.isalpha():
            corrected_words.append(word)
            continue
            
        start = max(0, i - context_window)
        end = min(len(words_in_text), i + context_window + 1)
        context_words = [w.lower() for w in words_in_text[start:end] if w.isalpha()]
        
        if word.lower() in word_set:
            corrected_words.append(word)
        else:
            corrected_word = spell_check(word.lower(), word_set, freq_dict, context_words)
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
        text_str = ' '.join(text_obj)
        
        sentences = sent_tokenize(text_str)
        
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Only include sentences with at least 3 words
                processed = process_text(sentence)[0]
                all_processed_sentences.append(processed)
                text_sources.append(text_names[idx])
    
    df = pd.DataFrame({
        'processed_text': all_processed_sentences,
        'source': text_sources
    })
    
    print(f"Total sentences processed: {len(df)}")
    return df

############################__PHASE 2__#############################
def prepare_dataset_for_classification(training_df):
    # Check if 'source' column exists (needed as target for classification)
    if 'source' not in training_df.columns:
        raise ValueError("Dataset must contain a 'source' column for classification")
    
    le = LabelEncoder()
    y = le.fit_transform(training_df['source'])
    
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"\nClass Labels: {label_mapping}")

    print("\nPerforming TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(training_df['processed_text'])
    print(f"Feature matrix shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, vectorizer, label_mapping

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_mapping):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    
    print("\nTraining and evaluating models:")
    
    for name, model in models.items():
        start_time = time.time()
        
        # Train the model
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        # Print summary
        print(f"{name} - Accuracy: {accuracy:.4f}, Training Time: {training_time:.2f} seconds")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
    
    # Print best model
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Detailed evaluation of best model
    y_pred = results[best_model_name]['predictions']
    print("\nDetailed evaluation of best model:")
    print(classification_report(y_test, y_pred, target_names=list(label_mapping.values())))
    
    return results, best_model_name, best_model

def save_model(model, filename):
    # function to save the model to a file
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    # function to load the model from a file
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Load data
    training_df = prepare_training_data()

    # Prepare dataset for classification
    X_train, X_test, y_train, y_test, vectorizer, label_mapping = prepare_dataset_for_classification(training_df)
    results, best_model_name, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test, label_mapping)

    # Save the best model
    save_model(best_model, "best_model.pkl")

    # Load the best model
    loaded_model = load_model("best_model.pkl")

#visualize the results
def visualize_results(results):
    # Create a bar plot of model accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=[result['accuracy'] for result in results.values()])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.tight_layout()
    plt.show()

    # Create a confusion matrix heatmap
    y_pred = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
