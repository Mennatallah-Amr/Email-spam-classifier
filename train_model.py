import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocess import preprocess_text
import pickle

df = pd.read_csv("spam_ham_dataset.csv")

df = df.drop(['Unnamed: 0', 'label'], axis=1)
df.rename(columns={'text': 'email', 'label_num': 'target'}, inplace=True)
df = df.drop_duplicates()

df["preprocess_email"] = df["email"].apply(preprocess_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["preprocess_email"]).toarray()
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model + vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model saved successfully.")
