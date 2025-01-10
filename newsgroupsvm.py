import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
X_text = newsgroups.data
y = newsgroups.target

# Convert the text data to a high-dimensional TF-IDF feature matrix
vectorizer = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features for manageability
X = vectorizer.fit_transform(X_text)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the model with SVM
svm = SVC(kernel="rbf", gamma=0.3, C=1.0)

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate and print the accuracy
accuracy_test = accuracy_score(y_test, y_pred)
print(f"Accuracy on the testing set: {accuracy_test * 100:.2f}%")

# Print the classification report for more detailed evaluation
print('Classification Report:')
print(classification_report(y_test, y_pred))

