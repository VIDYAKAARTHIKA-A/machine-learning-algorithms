'''1) import the necessary modules:
    numpy=for calculating using arrays
    pandas=to create dataframes
    sklearn learn.tree for decisiontreeclassifier
    sklearn.metrics= accuracy score
    '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
balance_data = pd.read_csv(r'C:/Users/Admin/Downloads/Crop_recommendation.csv', sep=',', header=0)

# Print dataset information- exploratory data analysis
print("Dataset length: ", len(balance_data))
print("Dataset shape: ", balance_data.shape)
print("Dataset preview:")
print(balance_data.head()) #viewing top 5 rows of the dataset
# in case the column names are numbers rename them
#select the feature variables and target variables
#features:sepal length, sepal width,petal length, petal width
#target variable=species
X = balance_data.iloc[:, 1:8].values  # Select columns 1 to 4 (features)
Y = balance_data.iloc[:, 9].values    # Select column 5 (target)

# Split the dataset into training and testing sets
#70%-training set
#30%-testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Initialize the Decision Tree Classifier and use entropy as the impurity measure
clf_en = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Train the model
clf_en.fit(X_train, y_train)

# Predict the labels for the train set
y_pred_train_en = clf_en.predict(X_train)
#predict the labels for test set
y_pred_test_en=clf_en.predict(X_test)
print("predictions using entropy train:",y_pred_train_en)
print("predictions using entropy test:",y_pred_test_en)
#checking accuracy for both training and testing data
accuracy_train_en=accuracy_score(y_train,y_pred_train_en)
print(f"Accuracy on the training set: {accuracy_train_en * 100:.2f}%")
accuracy_test_en = accuracy_score(y_test, y_pred_test_en)
print(f"Accuracy on the test set: {accuracy_test_en * 100:.2f}%")
#check for overfitiing or underfitting
print("Training set score: ",clf_en.score(X_train,y_train))
print("Testing set score: ", clf_en.score(X_test,y_test))
#Initialize with gini impurity
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3, min_samples_leaf=5)
#Train the model
clf_gini.fit(X_train,y_train)
y_pred_train_gini=clf_gini.predict(X_train)
y_pred_test_gini=clf_gini.predict(X_test)
print("predictions using gini train: ",y_pred_train_gini)
print("predicitons using gini test: ",y_pred_test_gini)
accuracy_train_gini=accuracy_score(y_train,y_pred_train_gini)
print(f"Accuracy on the training set: {accuracy_train_gini* 100:.2f}%")
accuracy_test_gini=accuracy_score(y_test, y_pred_test_gini)
print(f"Accuracy on the test set: {accuracy_test_gini * 100:.2f}%")

print("Training set score: ",clf_gini.score(X_train,y_train))
print("Testing set score: ", clf_gini.score(X_test,y_test))

from  sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test_en)
print("Confusion matrix:",cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test_en))
# Print the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_en, filled=True, feature_names=balance_data.columns[1:5], class_names=balance_data['Species'].unique())
plt.show()

