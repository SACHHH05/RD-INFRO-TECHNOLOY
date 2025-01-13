import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('wordnet')

file_path = '/content/sms spam.zip'  
data = pd.read_csv(file_path, encoding='latin-1')
data = data[['v1', 'v2']]  
data.columns = ['label', 'message']  


data['label'] = data['label'].map({'ham': 0, 'spam': 1})


sns.countplot(data['label'], palette='cool')
plt.title("Distribution of Ham vs Spam Messages")
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()


lemmatizer = WordNetLemmatizer()

def preprocess_message(message):
    message = re.sub(r'[^a-zA-Z\s]', '', message)
    message = message.lower()
    message = ' '.join([lemmatizer.lemmatize(word) for word in message.split()])
    return message

data['message'] = data['message'].apply(preprocess_message)

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear')
}

for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("-" * 60)

best_model = LogisticRegression()
best_model.fit(X_train_tfidf, y_train)
joblib.dump(best_model, 'spam_classifier_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

loaded_model = joblib.load('spam_classifier_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def classify_message(message):
    vectorized_message = loaded_vectorizer.transform([message])
    prediction = loaded_model.predict(vectorized_message)[0]
    return "Spam" if prediction == 1 else "Ham"

test_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Call now!",
    "Hey, are we still meeting at 5 pm today?",
    "Urgent! Your account has been compromised. Verify your details immediately.",
    "Can you send me the notes for the class?",
]

for msg in test_messages:
    print(f"Message: '{msg}' is classified as: {classify_message(msg)}")
