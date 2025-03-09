import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/KIIT/Desktop/AD Lab/Lab3/StudentPerformanceFactors.csv')

# Preview the dataset
print("Dataset Preview:")
print(data.head())

# Define features (X) and target (y)
X = data[['Hours_Studied', 'Attendance', 'Exam_Score']]
y = data['Exam_Score']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Calculate residuals
residuals = y_test - y_pred

# Define an anomaly threshold (e.g., 2 standard deviations from the mean residual)
threshold = 2 * np.std(residuals)

# Identify anomaly points
anomalies = X_test[np.abs(residuals) > threshold]
print("\nAnomaly Points:")
print(anomalies)

# Scatter plot: Actual vs Predicted, with anomalies highlighted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Normal Points')
plt.scatter(y_test[np.abs(residuals) > threshold], y_pred[np.abs(residuals) > threshold],
            color='red', alpha=0.8, label='Anomaly Points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--', label='Ideal Fit')
plt.title('Actual vs Predicted Performance (with Anomalies)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.show()
