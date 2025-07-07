# Week 4 Final Project - Complete Next Word Predictor using Transformers, Classical, and Embedding Models


import numpy as np
import pandas as pd
import nltk
import re
import torch
import gradio as gr

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer

nltk.download('punkt')

# Sample Dataset
corpus = [
    "I love machine learning",
    "Transformers are amazing",
    "Natural language processing is fun",
    "I enjoy studying deep learning",
    "Neural networks are powerful"
]
labels = [1, 1, 0, 1, 0]

# Preprocessing

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

corpus_clean = [preprocess(sent) for sent in corpus]

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus_clean)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nTF-IDF Classifier Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Load GloVe Embeddings

def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

glove_model = load_glove_model("glove.6B.300d.txt")  # Ensure this file exists in your path

# Sentence Vector Averaging

def get_average_vector(sentence, model, dim=300):
    tokens = word_tokenize(sentence.lower())
    vecs = [model[word] for word in tokens if word in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

sentence_vecs = np.array([get_average_vector(sent, glove_model) for sent in corpus_clean])
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(sentence_vecs, labels, test_size=0.2, random_state=42)
clf.fit(X_train_g, y_train_g)
y_pred_g = clf.predict(X_test_g)
print("\nGloVe Classifier Accuracy:", accuracy_score(y_test_g, y_pred_g))

# Hugging Face Sentiment Model
hf_model = pipeline("sentiment-analysis")
print("\nHugging Face Sentiment Predictions:")
for sent in corpus:
    print(f"{sent} -> {hf_model(sent)}")

# GPT-2 Next Word Prediction

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def predict_next_words(text, k=5):
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    top_k = torch.topk(next_token_logits, k)
    predicted_tokens = [gpt2_tokenizer.decode([idx]) for idx in top_k.indices]
    return predicted_tokens

sample_prompt = "Deep learning models are"
print(f"\nPrompt: {sample_prompt}")
print("Top-5 Predictions:", predict_next_words(sample_prompt))

# Gradio App for GPT-2 Completion

def gpt2_completion(prompt):
    tokens = predict_next_words(prompt, k=5)
    return "\n".join(tokens)

grad_interface = gr.Interface(
    fn=gpt2_completion,
    inputs=gr.Textbox(label="Enter Prompt"),
    outputs=gr.Textbox(label="Top-5 Next Word Predictions"),
    title="GPT-2 Next Word Predictor"
)

# Uncomment below line to launch the app
# grad_interface.launch()
