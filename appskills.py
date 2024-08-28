import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
crop = pd.read_csv("Crop_recommendation.csv")
print(crop.head())

# Select only the numeric columns
numeric_df = crop.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr = numeric_df.corr()

# Plot the heatmap
sns.heatmap(corr, annot=True, cbar=True, cmap='coolwarm')

# Show the plot
sns.displot(crop['N'])
plt.show()
