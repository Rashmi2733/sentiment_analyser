import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle


df = pd.read_csv('Tweets.csv')
df.head()
# Converting text to lowercase and string type
df['text'] = df['text'].str.lower().astype(str)  
X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


#Vectorizing the text before fitting in the model
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


#Fitting and training SVM model
model = SVC().fit(X_train_vectors, y_train)
y_pred = model.predict(X_test_vectors)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))


with open('svm_sentiment_analyser.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)