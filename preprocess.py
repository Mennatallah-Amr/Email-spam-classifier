import nltk
import string
import os
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)


# Initialize tools

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


# Preprocessing function

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)
