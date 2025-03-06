import seaborn as sns
import pandas as pd
import numpy as np

# create scatter plot to understand the relation between data features using seaborn.scatterplot
import matplotlib.pyplot as plt

# Generate a sample DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
})
# Assuming you have a DataFrame named df and you want to plot 'feature1' vs 'feature2'
sns.scatterplot(data=df, x='feature1', y='feature2')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of Feature 1 vs Feature 2')
plt.show()