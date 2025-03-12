# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset (Using synthetic dataset for demonstration)
# Create a synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,   # Number of features
    n_informative=15, # Number of informative features
    n_redundant=5,    # Number of redundant features
    n_classes=2,      # Binary classification
    random_state=42
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(1, 21)])
df['Spam'] = y  # Target variable (0: Not Spam, 1: Spam)

# Step 2: Split the dataset
X = df.drop('Spam', axis=1)
y = df['Spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
print("\nClassification Report:")
print(class_report)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
