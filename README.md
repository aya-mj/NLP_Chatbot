# NLP_Chatbot

## Project Overview
This project aims to create a natural language processing chatbot using NLTK for text classification. The chatbot will be able to classify text into different categories such as "technical support," "billing," or "shipping."

## Phase 1: Text Processing and Spelling Correction
Phase 1 focuses on implementing fundamental NLP techniques for text processing and spelling correction using Python's Natural Language Toolkit (NLTK) and scikit-learn.

### Features Implemented

#### Text Processing
- Text cleaning (lowercase conversion, special character removal)
- Tokenization (word and sentence level)
- Stopword removal, Stemming 
- Pipeline creation for standardized text preprocessing
- Spelling Correction ()

#### Performance Measurement
- Performance monitoring using timing decorators
- Optimized suggestion generation with limits

#### Training Data Preparation
- Sentence extraction from NLTK book corpus
- Processing of multiple text sources
- Creation of training dataset with source tracking




### Dependencies
- nltk
- scikit-learn
- pandas
- numpy
- re
- collections
- time

### Next Steps
Phase 2 will focus on implementing the text classification model to categorize text into different service categories (technical support, billing, shipping, etc.).

## Getting Started
1. Install required dependencies: `pip install nltk scikit-learn pandas numpy`
2. Uncomment the NLTK download lines if you're running for the first time
3. Import the module and use the functions as shown in the usage example