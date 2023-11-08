import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np  
from read_merged_csv import read_merged

# Assuming `df` is your DataFrame and it has a binary column 'InMichelin' where 1 indicates a Michelin-starred restaurant.
data = read_merged()  # Make sure this function returns the DataFrame correctly
data.fillna(data.mean(), inplace=True)

# Group the data by Michelin status and use the precalculated 'value' and 'excellent_to_terrible_ratio'
grouped_data = data.groupby('InMichelin').mean().reset_index()

def make_radar_plot(grouped_data, title, categories):
    # Number of variables we're plotting.
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The radar chart is circular so we need to complete the loop
    values = grouped_data[categories].values.tolist()
    values = [v + v[:1] for v in values]  # Repeat the first value to close the circle
    angles += angles[:1]  # Complete the loop

    # Draw the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)
    plt.ylim(0, max(grouped_data[categories].max()))

    # Plot data
    for value in values:
        ax.plot(angles, value, linewidth=2, linestyle='solid')
        ax.fill(angles, value, alpha=0.25)

    # Add a title and legend
    plt.title(title, size=20, color='grey', y=1.1)
    labels = ('Non-Michelin', 'Michelin')
    plt.legend(labels, loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the plot
    plt.show()

# Select the categories you want to include in the radar plot
categories = ['food', 'service', 'price', 'decor', 'excellent_to_terrible_ratio', 'value']

# Normalize the data for the plot (so that the radar chart is scaled to the maximum of each variable)
normalized_data = grouped_data.copy()
for col in categories:
    max_value = normalized_data[col].max()
    normalized_data[col] = normalized_data[col] / max_value * 50  # Scale to 50 for the radar chart

make_radar_plot(normalized_data, 'Michelin vs Non-Michelin Restaurant Comparison', categories)
