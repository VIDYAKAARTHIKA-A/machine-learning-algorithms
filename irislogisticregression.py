import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
iris = load_iris()
X = iris.data  # All features are used
y = iris.target  # Target variable
target_names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(multi_class="multinomial", solver='lbfgs')  # Specify multi-class

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Display Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores: ", cv_scores)
print("Mean CV Accuracy: {:.2f}%".format(cv_scores.mean() * 100))
# Visualize the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 1], y=X_test[:, 2], hue=y_test, palette='Set1', marker='o', legend='full')
plt.xlabel("Sepal Width")
plt.ylabel("Petal Length")
plt.title("Iris Test Data Scatter Plot\n(Colored by Actual Species)")
plt.legend(title="Species", loc="upper right", labels=target_names)
plt.show()
