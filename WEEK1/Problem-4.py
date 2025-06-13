import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



positive_reviews = ["Good", "Great", "Marvellous", "Loved it", "Fantastic", "Nice", "Excellent Movie", "Thrilling", "Funny", "Furious"]* 5
negative_reviews = ["Bad", "Worst", "Terrible", "Hated it", "Boring", "Awful", "Not Recommended", "Not nice", "Too slow", "Not funny"]* 5

rewiews = positive_reviews+ negative_reviews
sentiments = ['positive']*50 + ['negative']*50

df = pd.DataFrame({'Review' : rewiews, 'Sentiment' : sentiments})

vectorizer = CountVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(df['Review'])

X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    return prediction[0]

print(predict_review_sentiment(model, vectorizer, "This movie was fantastic!"))
