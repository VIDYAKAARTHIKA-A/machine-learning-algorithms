import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
X = iris.data[:, :2]  # Select the first two features for visualization
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the SVM model with specified hyperparameters
svm = SVC(kernel="linear", C=0.2)  # 'gamma' is not used in linear kernel

# Train the model using the training data
svm.fit(X_train, y_train)

# Predict on the testing data
y_pred = svm.predict(X_test)

# Calculate the accuracy on the testing set
accuracy_test = accuracy_score(y_test, y_pred)
print(f"Accuracy on the testing set: {accuracy_test * 100:.2f}%")

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    cmap=plt.cm.Spectral,
    alpha=0.8
)

# Scatter plot of the data points
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k", label='Data Points')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('SVM Decision Boundary and Iris Data Points')
plt.legend()
plt.show()
