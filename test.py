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
from sklearn.cluster import KMeans

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
# Text Classification with Multiple Models and Model Evaluation

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import pickle

def prepare_dataset_for_classification(training_df):
    """
    Prepare processed text data for classification
    
    Args:
        training_df: DataFrame with processed text and source labels
        
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    print("\n" + "="*50)
    print("PHASE 2: Text Classification Model Training")
    print("="*50)
    
    # Check if 'source' column exists (needed as target for classification)
    if 'source' not in training_df.columns:
        raise ValueError("Dataset must contain a 'source' column for classification")
    
    # Convert source names to numeric labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(training_df['source'])
    
    # Store label mapping for interpretation
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"\nClass Labels: {label_mapping}")
    
    # Convert text to TF-IDF features
    print("\nPerforming TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(training_df['processed_text'])
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, vectorizer, label_mapping

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_mapping):
    """
    Train and evaluate multiple classification models
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        label_mapping: Dictionary mapping numeric labels to text classes
        
    Returns:
        Dictionary with trained models and their performance metrics
    """
    # Define models to evaluate
    models = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    
    # Train and evaluate each model
    print("\nTraining and evaluating models:")
    print("-"*50)
    
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

def perform_cross_validation(models, X, y, cv=5):
    """
    Perform cross-validation for each model
    
    Args:
        models: Dictionary of models to evaluate
        X: Feature matrix
        y: Target labels
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation scores
    """
    print("\nPerforming cross-validation...")
    cv_results = {}
    
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_results[name] = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
        print(f"{name} - Mean CV Accuracy: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
    return cv_results

def hyperparameter_tuning(X_train, y_train, best_model_name):
    """
    Perform hyperparameter tuning for the best model
    
    Args:
        X_train: Training feature matrix
        y_train: Training target labels
        best_model_name: Name of the best model to tune
        
    Returns:
        Best model after tuning
    """
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    # Define parameter grids for different models
    param_grids = {
        'Naive Bayes': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
    }
    
    # Select model class based on best model name
    model_classes = {
        'Naive Bayes': MultinomialNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Get appropriate model and parameter grid
    model = model_classes[best_model_name]
    param_grid = param_grids[best_model_name]
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Print results
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    return grid_search.best_estimator_

def visualize_model_comparison(results, cv_results=None):
    """
    Visualize model comparison results
    
    Args:
        results: Dictionary with model results
        cv_results: Dictionary with cross-validation results
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # Prepare data
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        train_times = [results[name]['training_time'] for name in model_names]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy comparison
        sns.barplot(x=model_names, y=accuracies, palette='viridis', ax=ax1)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Model')
        ax1.set_ylim(0, 1)
        
        # Add values on top of bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f"{v:.4f}", ha='center')
            
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot training time comparison
        sns.barplot(x=model_names, y=train_times, palette='viridis', ax=ax2)
        ax2.set_title('Model Training Time Comparison')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_xlabel('Model')
        
        # Add values on top of bars
        for i, v in enumerate(train_times):
            ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center')
            
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("\nModel comparison visualization saved to 'model_comparison.png'")
        
        # If CV results are available, plot them too
        if cv_results:
            plt.figure(figsize=(10, 6))
            
            cv_means = [cv_results[name]['mean_score'] for name in model_names]
            cv_stds = [cv_results[name]['std_score'] for name in model_names]
            
            # Create bar plot with error bars
            bars = plt.bar(model_names, cv_means, yerr=cv_stds, alpha=0.8, capsize=10)
            
            plt.title('Cross-Validation Results')
            plt.ylabel('Mean Accuracy')
            plt.xlabel('Model')
            plt.ylim(0, 1)
            
            # Add values on top of bars
            for i, v in enumerate(cv_means):
                plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
                
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('cross_validation_results.png')
            print("Cross-validation results visualization saved to 'cross_validation_results.png'")
        
    except ImportError:
        print("Matplotlib or Seaborn not installed. Skipping visualization.")

def visualize_confusion_matrix(y_test, y_pred, label_mapping):
    """
    Visualize confusion matrix for best model
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        label_mapping: Dictionary mapping numeric labels to text classes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_mapping.values(),
                    yticklabels=label_mapping.values())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix visualization saved to 'confusion_matrix.png'")
        
    except ImportError:
        print("Matplotlib or Seaborn not installed. Skipping visualization.")

def classify_new_text(text, vectorizer, model, label_mapping):
    """
    Classify new text using the best trained model
    
    Args:
        text: Text to classify
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classification model
        label_mapping: Dictionary mapping numeric labels to text classes
        
    Returns:
        Dictionary with classification results
    """
    # Process text first (using Phase 1 functionality)
    processed_text = process_text(text)[0]
    
    # Apply spell correction
    corrected_text = correct_spelling(text, dictionary, word_freq_dict)
    
    # Vectorize the corrected text
    X_new = vectorizer.transform([corrected_text])
    
    # Predict class
    predicted_class = model.predict(X_new)[0]
    class_name = label_mapping[predicted_class]
    
    # Get probability scores if model supports it
    try:
        probabilities = model.predict_proba(X_new)[0]
        class_probs = {label_mapping[i]: prob for i, prob in enumerate(probabilities)}
    except:
        class_probs = None
    
    print(f"\nText Classification Results:")
    print(f"Original text: {text}")
    print(f"Corrected text: {corrected_text}")
    print(f"Predicted class: {class_name}")
    
    if class_probs:
        print("Class probabilities:")
        for class_name, prob in sorted(class_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"- {class_name}: {prob:.4f}")
    
    return {
        'original_text': text,
        'processed_text': processed_text,
        'corrected_text': corrected_text,
        'predicted_class': predicted_class,
        'class_name': class_name,
        'probabilities': class_probs
    }

def save_model(model, vectorizer, label_mapping, output_dir='./model'):
    """
    Save trained model, vectorizer, and label mapping
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        label_mapping: Dictionary mapping numeric labels to text classes
        output_dir: Directory to save model files
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    with open(f"{output_dir}/text_classifier.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    with open(f"{output_dir}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save label mapping
    with open(f"{output_dir}/label_mapping.pkl", 'wb') as f:
        pickle.dump(label_mapping, f)
    
    print(f"\nModel saved to {output_dir}/")

def load_model(model_dir='./model'):
    """
    Load saved model, vectorizer, and label mapping
    
    Args:
        model_dir: Directory with model files
        
    Returns:
        Loaded model, vectorizer, and label mapping
    """
    # Load model
    with open(f"{model_dir}/text_classifier.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Load vectorizer
    with open(f"{model_dir}/tfidf_vectorizer.pkl", 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load label mapping
    with open(f"{model_dir}/label_mapping.pkl", 'rb') as f:
        label_mapping = pickle.load(f)
    
    return model, vectorizer, label_mapping

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("NLP Text Analysis Pipeline - Phase 1 & 2")
    print("="*50)
    
    # Phase 1: Prepare training data (reusing from original code)
    print("\nPHASE 1: Text Processing & Spell Checking")
    training_df = prepare_training_data()
    
    # Phase 2: Text Classification Model Training and Evaluation
    try:
        # Prepare data for classification
        X_train, X_test, y_train, y_test, vectorizer, label_mapping = prepare_dataset_for_classification(training_df)
        
        # Train and evaluate models
        results, best_model_name, best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test, label_mapping)
        
        # Perform cross-validation
        cv_results = perform_cross_validation({name: results[name]['model'] for name in results}, X_train, y_train)
        
        # Tune hyperparameters for best model
        tuned_model = hyperparameter_tuning(X_train, y_train, best_model_name)
        
        # Evaluate tuned model
        tuned_y_pred = tuned_model.predict(X_test)
        tuned_accuracy = accuracy_score(y_test, tuned_y_pred)
        print(f"\nTuned model accuracy: {tuned_accuracy:.4f}")
        print("Classification report for tuned model:")
        print(classification_report(y_test, tuned_y_pred, target_names=list(label_mapping.values())))
        
        # Visualize results
        visualize_model_comparison(results, cv_results)
        visualize_confusion_matrix(y_test, tuned_y_pred, label_mapping)
        
        # Save best model
        save_model(tuned_model, vectorizer, label_mapping)
        
        # Test with example text
        print("\n" + "="*50)
        print("Testing with example text:")
        
        input_text = "Thy shall find the meaning in the words of wisdom."
        classification = classify_new_text(input_text, vectorizer, tuned_model, label_mapping)
        
        print("\n" + "="*50)
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error in Phase 2: {str(e)}")

# # Example usage
# if __name__ == "__main__":
#     training_df = prepare_training_data()

#     # Convert training data to TF-IDF features
#     vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#     X = vectorizer.fit_transform(training_df['processed_text'])
    
#     # Train KMeans clustering
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     kmeans.fit(X)
    
#     # Add cluster labels to training data
#     training_df['cluster'] = kmeans.labels_

#     # Save training data with clusters
#     training_df.to_csv('training_data_with_clusters.csv', index=False)
#     print("Training data with clusters saved to 'training_data_with_clusters.csv'")

#     # Example usage of spell check
#     input_text = "Thiss is a smple sentnce with soome mistkes. We ned to corect it."
#     corrected = correct_spelling(input_text, dictionary, word_freq_dict)

#     print("Corrected Sentence:")
#     print(corrected)

        