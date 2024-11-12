from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

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
