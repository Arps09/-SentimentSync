# 📈 SentimentSync – Amazon Review Sentiment Analyzer

**SentimentSync** is a web-based Sentiment Analysis app built using **Streamlit**. It allows users to input product reviews (e.g., from Amazon), and uses a machine learning model to predict whether the review is **Positive** or **Negative**.

---

## 🧠 What is Sentiment Analysis?

Sentiment analysis is a technique in **Natural Language Processing (NLP)** that determines whether a piece of text expresses a **positive**, **negative**, or **neutral** emotion.

With the explosion of user-generated content like reviews, tweets, and comments, analyzing sentiment helps:

- 🧑‍💻 Customers make informed decisions
- 📈 Businesses track brand/product perception
- 📊 Researchers understand public opinion

---

## 🔍 What Problem Does SentimentSync Solve?

Platforms like Amazon get **millions of reviews**, making it impossible to manually monitor sentiment. SentimentSync:

- Automatically classifies customer reviews
- Helps businesses gauge user satisfaction
- Enables real-time feedback analysis

---

## ⚙️ How It Works

### 📝 1. Text Input
User enters a product review into the app.

### 🧹 2. Text Preprocessing
The text is cleaned using:
- Lowercasing
- Removal of HTML tags, URLs, punctuation
- Stopword removal
- **Lemmatization** with NLTK

### 📊 3. Feature Extraction
Text is converted into numeric vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### 🧠 4. Sentiment Prediction
A trained **Logistic Regression** model analyzes the vector and classifies the sentiment.

### 🎯 5. Output
The app displays either:
- 😊 **Positive**
- 😠 **Negative**

---

## 🖼️ Features

- Live sentiment analysis via text input
- Clean and modern Streamlit interface
- Lightweight and fast
- TF-IDF + Logistic Regression combo
- Extensible for other NLP projects

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- pip
 
 ---
 
## 📦 Requirements
- txt
- Copy
- Edit
- streamlit
- pandas
- numpy
- nltk
- scikit-learn

## 🧪 Model Info
- Vectorizer: TF-IDF

- Classifier: Logistic Regression

- Libraries: scikit-learn, NLTK

- Training: Done in proj.ipynb

## 👤 Author
- Made with ❤️ by Arpita Mishra
- If you like this project, consider giving it a ⭐!






