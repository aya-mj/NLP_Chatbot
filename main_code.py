import nltk
from nltk.book import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer


# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('book')

# Set up stop words
stop_words = set(stopwords.words('english'))

# Function to clean text: lowercase, remove non-alphabetic characters, and stopwords
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()  # Split text into words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Example usage
text1_sample = ' '.join(text1[:5000])  
cleaned_text1 = clean_text(text1_sample) 

#tokenize the text
words = word_tokenize(cleaned_text1)
print(words[:10])  # Print first 10 words

# #tokenize the text into sentences
# sentences = sent_tokenize(cleaned_text1)
# print(sentences[0])  # Print first 10 sentences


stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words:", stemmed_words[:10])  # Print first 10 stemmed words



