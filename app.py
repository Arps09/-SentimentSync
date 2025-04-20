import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="SentimentSync", page_icon="📈", layout="centered")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("logistic_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# Streamlit UI
st.title("📈SentimentSync")
st.write("Please enter an review on our product on Amazon")

user_input = st.text_area("📝 Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        sentiment = "Positive 😊" if str(prediction).lower() == "positive" else "Negative 😠"
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.write("**Original Review:**")
        st.write(user_input)
