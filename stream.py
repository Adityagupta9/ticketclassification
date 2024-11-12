import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import pickle

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Custom transformer for text preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self.pre_process(text) for text in X]
    
    def pre_process(self, text):
        # Tokenize, remove stop words and punctuation, apply stemming
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words and word not in string.punctuation]
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        return ' '.join(stemmed_tokens).lower()

# Load the pipeline model
def load_model(file_path):
    try:
        # Attempt to load with joblib
        return joblib.load(file_path)
    except Exception as e:
        # Attempt loading with pickle as a fallback for version compatibility
        st.warning("Loading model with joblib failed. Trying pickle as an alternative.")
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file, encoding="latin1")
        except Exception as e:
            st.error("Model loading failed. Please check the model file.")
            return None

model = load_model("pipeline.joblib")  # Replace with your actual model file path

# Streamlit app
st.title("Support Ticket Classification")
st.write("This app classifies support ticket inquiries into categories.")

# Input text area for user input
user_input = st.text_area("Enter the support ticket inquiry:")

# Predict button
if st.button("Classify Inquiry"):
    if model:
        if user_input.strip():  # Ensure input is not empty
            try:
                # Use the pipeline to predict
                prediction = model.predict([user_input])
                st.success(f"Predicted Category: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter a valid inquiry to classify.")
    else:
        st.error("Model could not be loaded. Please check the model file.")
