import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
balance_data = pd.read_csv(r'C:\Users\Admin\Downloads\Iris.csv', sep=',', header=0)

# Display shape and head of the data
print(balance_data.shape)
print(balance_data.head())

# Correct feature and target selection
X = balance_data.drop(['Species', 'Id'], axis=1)  # Drop 'Species' (target) and 'Id' (if present)
y = balance_data['Species']  # Target

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)

# Fit the model
rfc.fit(X_train, y_train)

# Predict on test set
y_pred = rfc.predict(X_test)

# Check accuracy
print("Accuracy with default n_estimators:", accuracy_score(y_test, y_pred))

# Initialize RandomForestClassifier with 100 trees
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
y_pred_100 = rfc_100.predict(X_test)
print("Accuracy with 100 n_estimators:", accuracy_score(y_test, y_pred_100))

# Using a single RandomForestClassifier instance for feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Feature importance
column_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
feature_scores = pd.Series(clf.feature_importances_, index=column_names).sort_values(ascending=False)

# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

# Add title to the graph
plt.title("Visualizing Important Features")

# Visualize the graph
plt.show()

# Checking performance after removing 'SepalWidthCm' as an example
X_reduced = balance_data.drop(['Species', 'Id', 'SepalWidthCm'], axis=1)  # Remove 'SepalWidthCm'

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.33, random_state=42)

clf_reduced = RandomForestClassifier(random_state=0)
clf_reduced.fit(X_train_red, y_train_red)

# Predict on the reduced feature set
y_pred_reduced = clf_reduced.predict(X_test_red)

# Check accuracy score
print('Model accuracy score with SepalWidthCm removed: {0:0.4f}'.format(accuracy_score(y_test_red, y_pred_reduced)))

# Print classification report and confusion matrix
print(classification_report(y_test_red, y_pred_reduced))
print("Confusion Matrix:\n", confusion_matrix(y_test_red, y_pred_reduced))
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores: ", cv_scores)
print("Mean CV Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

