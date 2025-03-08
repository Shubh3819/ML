import seaborn as sns
import pandas as pd
import numpy as np

# use corelation heat map to find the relationship between the multiple features within the dataset using seaborns heatmap
import matplotlib.pyplot as plt

# Load the dataset
# data = pd.read_csv('/run/media/psycho/Windows-SSD/Project/KIIT/AD/lab1/dataset.csv')
# Generate a custom dataset
np.random.seed(0)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'feature4': np.random.rand(100)
})

# Calculate the correlation matrix
corr = data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
