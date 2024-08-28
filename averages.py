import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Sample Data for Crop Predictions
crops = ['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
performance = [75, 96, 85, 64, 78, 11, 65, 91, 74, 92, 13, 53, 71, 79, 15, 20, 54, 48, 85, 32, 82]

# Debugging: Print the lengths of the lists
print(f"Length of 'crops': {len(crops)}")
print(f"Length of 'performance': {len(performance)}")

# Ensure all arrays are of the same length
if len(crops) != len(performance):
    raise ValueError("All arrays must be of the same length")

# Create a DataFrame
data = pd.DataFrame({
    'Crop': crops,
    'Average_Performance': performance
})

# Create a figure and axis for the flowchart
fig, ax = plt.subplots(figsize=(7, 5))  # Reduced the figure size

# Define boxes and arrows for the flowchart
boxes = {
    'Load and Preprocess Data': (0.1, 0.7, 0.3, 0.1),
    'Train and Save Model': (0.1, 0.5, 0.3, 0.1),
    'Flask Application Setup': (0.1, 0.3, 0.3, 0.1),
    'HTML Form Interaction': (0.1, 0.1, 0.3, 0.1)
}

# Add boxes to the plot
for label, (x, y, width, height) in boxes.items():
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05", edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    plt.text(x + width / 2, y + height / 2, label, horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')

# Add arrows
arrows = [
    ((0.25, 0.7), (0.25, 0.6)),
    ((0.25, 0.5), (0.25, 0.4)),
    ((0.25, 0.3), (0.25, 0.2))
]

for (start, end) in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor='black', shrink=0.05))

# Set limits and hide axes
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.8)
ax.axis('off')

# Title for the flowchart
plt.title('Workflow of Flask Application and Machine Learning Model', fontsize=14, weight='bold')

# Save the flowchart
plt.savefig('workflow_flowchart.png')
plt.show()

# Create a figure and axis for the crop performance bar chart
fig, ax = plt.subplots(figsize=(6, 4))  # Reduced the figure size

# Plot the bar chart for crop performance
bars = ax.bar(data['Crop'], data['Average_Performance'], color='skyblue')

# Add labels and title
ax.set_xlabel('Crop')
ax.set_ylabel('Average Performance (%)')
ax.set_title('Average Performance of Crops Based on Predictions', fontsize=16, weight='bold')

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=12)

# Save the bar chart
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('crop_performance.png')
plt.show()
