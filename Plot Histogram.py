import numpy as np

# plot the distribution using matplot library
import matplotlib.pyplot as plt

# Generate some data
data = np.random.random(100)

# Create a histogram
plt.hist(data, bins=100, edgecolor='black')

# Add title and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Save the plot to a file
# plt.savefig('data_distribution.png')

plt.show()
