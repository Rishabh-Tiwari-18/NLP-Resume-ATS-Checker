import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK Data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\\S+', ' ', text)

    # Remove mentions
    text = re.sub(r'@\\S+', ' ', text)

    # Remove hashtags
    text = re.sub(r'#\\S+', ' ', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\\s+', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)