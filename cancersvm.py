# Load the important packages
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load the datasets
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target
 
#Build the model
svm = SVC(kernel="rbf", gamma=0.3, C=1.0)
'''gamma= defines influence of single training example. higher gamma= make decision boundary
tight  and leads to overfitting'''
'''C regularization parameter- higher C- less regularization. lower C- more regularization'''
# Trained the model
svm.fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_pred=svm.predict(X_test)
accuracy_test=accuracy_score(y_test,y_pred)
print(f"Accuracy on the testing set: {accuracy_test* 100:.2f}%")

 
# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,
        xlabel=cancer.feature_names[0],
        ylabel=cancer.feature_names[1],
    )
 
# Scatter plot
plt.scatter(X[:, 0], X[:, 1], 
            c=y, 
            s=20, edgecolors="k")
plt.show()
