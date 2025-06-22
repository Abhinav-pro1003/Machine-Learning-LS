import pandas as pd
import numpy as np
import nltk
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

word2vec = api.load('word2vec-google-news-300')

df = pd.read_csv("Tweets.csv", encoding="latin-1")[["airline_sentiment","text"]]
df.columns = ["target","tweet content"]

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = contractions.fix(text) 
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+","",text)
    text = re.sub(r"[^\w\s]","",text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return tokens

df['tokens'] = df['tweet content'].apply(preprocess)

def vectorize(tokens):
    vectors = [word2vec[w] for w in tokens if w in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

df['features'] = df['tokens'].apply(vectorize)

X = np.vstack(df['features'].values)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess(tweet)
    vec = vectorize(tokens)
    return model.predict([vec])[0]

print(predict_tweet_sentiment(classifier, word2vec, "Delayed again! Very frustrated with this airline."))
