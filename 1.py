import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

import matplotlib.pyplot as plt

Dataset_size = 3000
# Load the California housing dataset
housing = fetch_california_housing()
housing.data = housing.data[-Dataset_size:]
housing.target = housing.target[-Dataset_size:]

# for x in housing:
#     print(x)
#     print(housing[x])

# Select features and target variable
X = housing.data[:, [0, 2, 4]]  # Features
y = housing.target  # Target is the median house value

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Visualize the regression line
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
plt.xlabel('Median Income ($100,000)')
plt.ylabel('Median House Value ($100,000)')
plt.title('Linear Regression - California Housing Prices')

#display mse and r2 in graph itself
plt.text(0.1, 0.9, f"MSE: {mse:.2f} ", fontsize=12, transform=plt.gcf().transFigure)
plt.text(0.19, 0.9, f"R2 Score: {r2:.2f}", fontsize=12, transform=plt.gcf().transFigure)

# Plot the regression line
line_x = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
line_y = model.predict(np.column_stack((line_x, np.full_like(line_x, X_test[:, 1].mean()), np.full_like(line_x, X_test[:, 2].mean()))))
plt.plot(line_x, line_y, color='green', label='Regression Line')

# sort_axis = operator.itemgetter(0)
# sorted_zip = sorted(zip(X_test[:, 0], y_pred), key=sort_axis)
# X_test[:, 0], y_pred = zip(*sorted_zip)
# plt.plot(X_test[:, 0], y_pred, color='green', label='Regression Line')

plt.legend()
plt.show()


# :Attribute Information:
#     0- MedInc        median income in block group
#     1- HouseAge      median house age in block group
#     2- AveRooms      average number of rooms per household
#     3- AveBedrms     average number of bedrooms per household
#     4- Population    block group population
#     5- AveOccup      average number of household members
#     6- Latitude      block group latitude
#     7- Longitude     block group longitude