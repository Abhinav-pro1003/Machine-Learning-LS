import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api

word2vec = api.load('word2vec-google-news-300') 
# downloads ~1.6GB model

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
df.columns = ["Labels","Message"]

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df['tokens'] = df['Message'].apply(preprocess)

def vectorize(tokens):
    vectors = [word2vec[w] for w in tokens if w in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

df['features'] = df['tokens'].apply(vectorize)

X = np.vstack(df['features'].values)
y = df['Labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

def predict_message_class(classifier, w2v_model, message):
    tokens = preprocess(message)
    vec = vectorize(tokens)
    return classifier.predict([vec])[0]

print("Predict_message_class", predict_message_class(classifier, word2vec, "Ok lar... Joking wif u oni...,,,"))