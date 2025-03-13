# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
target_names = iris.target_names  # Class names

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)  # Create the model
clf.fit(X_train, y_train)  # Train the model

# Step 4: Make predictions
y_pred = clf.predict(X_test)

# Step 5: Evaluate the model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=target_names)
print("\nClassification Report:")
print(class_report)

# Step 6: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=target_names, filled=True, rounded=True)
plt.title("Decision Tree for Iris Classification")
plt.show()
