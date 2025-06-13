import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

good_reviews = ["Good", "Nice", "Loved it", "Amazing", "Highly recommended"]*10
bad_reviews = ["Bad", "Hated it", "Worst", "Terriblr", "Not useful"]*10

reviews = good_reviews + bad_reviews
labels = ['good']*50 + ['bad']*50
data = ({'Reviews' : reviews, 'Labels' : labels})

df=pd.DataFrame(data)
vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['Reviews'])

X_train, X_test, Y_train, Y_test = train_test_split(X, df['Labels'], test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(X_train, Y_train)

pred = model.predict(X_test)
print(classification_report(Y_test, pred))

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

new_test = ["Loved it"]
vectorized_text = text_preprocess_vectorize(new_test, vectorizer)
print(model.predict(vectorized_text))
