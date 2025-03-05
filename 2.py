import pandas as pd
from sklearn.datasets import load_iris

# load a sample data set using pandas( for e.g. iris data set and custom data set)
# Load iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Load custom dataset
custom_data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [10, 20, 30, 40, 50]
}
custom_df = pd.DataFrame(custom_data)

print("Iris DataFrame:")
print(iris_df.head())
print("\nCustom DataFrame:")
print(custom_df.head())

# print(iris_df[['sepal length (cm)']].head())