#  SMS SPAM
---
Overview:
---
This project implements an SMS spam detection system using Natural Language Processing (NLP) techniques and machine learning models. The goal is to classify SMS messages as either Ham (legitimate) or Spam.
---
Features:
---
Preprocessing of SMS messages including cleaning, normalization, and lemmatization.

Feature extraction using TF-IDF Vectorization.

Implementation and evaluation of three machine learning models:

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (SVM)

Visualization of the dataset distribution (Ham vs Spam).

Saving the best model and vectorizer using Joblib for future use.

A function to classify new SMS messages as Ham or Spam.
---
Project Structure:
---
sms_spam.zip: Contains the dataset used for training and testing.

spam_classifier_model.pkl: Saved Logistic Regression model.

tfidf_vectorizer.pkl: Saved TF-IDF vectorizer.

Python script implementing the project.
---
Dataset:
---
The dataset used in this project is a collection of SMS messages labeled as Ham or Spam. It is processed as follows:

Removal of non-alphabetic characters.

Conversion to lowercase.

Lemmatization of words.

The processed data is split into training and testing sets using an 80-20 split.
---
Libraries Used:
---
pandas: Data manipulation and analysis.

re: Regular expressions for text preprocessing.

nltk: Natural Language Toolkit for lemmatization.

scikit-learn: Machine learning model implementation and evaluation.

joblib: Model serialization and deserialization.

matplotlib & seaborn: Data visualization.
---
How to Run:
---
Install required libraries:

pip install pandas numpy nltk scikit-learn joblib matplotlib seaborn

Download and unzip the dataset to the specified path (/content/sms spam.zip).

Run the Python script.

Use the saved model and vectorizer to classify new messages using the classify_message function.
---
Results:
---
Logistic Regression achieved the best performance among the tested models.

Detailed classification reports and accuracy scores are printed for each model.
---
Visualization:
---
The dataset distribution is visualized using a bar chart:

Ham messages: Count of legitimate messages.

Spam messages: Count of spam messages.
---
Future Enhancements:
---
Experiment with other machine learning algorithms.

Include more advanced preprocessing techniques like stemming.

Add a web or mobile interface for real-time spam detection.

Use deep learning models for improved accuracy.
