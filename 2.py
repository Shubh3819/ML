# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a synthetic dataset
# Features: Study Hours, Class Attendance, Assignment Scores
# Target: Final Exam Score
np.random.seed(42)  # For reproducibility
n_samples = 200
study_hours = np.random.uniform(1, 10, n_samples)
class_attendance = np.random.uniform(50, 100, n_samples)
assignment_scores = np.random.uniform(20, 100, n_samples)

# Target variable (Final Exam Score)
# Adding some noise for realism
exam_scores = (
    5 * study_hours +
    0.3 * class_attendance +
    0.5 * assignment_scores +
    np.random.normal(0, 5, n_samples)  # Adding noise
)

# Create a DataFrame
data = pd.DataFrame({
    'Study Hours': study_hours,
    'Class Attendance': class_attendance,
    'Assignment Scores': assignment_scores,
    'Exam Scores': exam_scores
})

# Step 2: Split the data into training and testing sets
X = data[['Study Hours', 'Class Attendance', 'Assignment Scores']]
y = data['Exam Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 6: Scatter plot of Actual vs Predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='-', linewidth=2)
plt.title('Actual vs Predicted Exam Scores')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.grid(True)
plt.show()
