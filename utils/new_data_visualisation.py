import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np  
from read_merged_csv import read_merged


data = read_merged() 
data.fillna(data.mean(), inplace=True)
grouped_data = data.groupby('InMichelin').mean().reset_index()

def make_radar_plot(grouped_data, title, categories):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values = grouped_data[categories].values.tolist()
    values = [v + v[:1] for v in values]
    angles += angles[:1]  
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)
    plt.ylim(0, max(grouped_data[categories].max()))
    for value in values:
        ax.plot(angles, value, linewidth=2, linestyle='solid')
        ax.fill(angles, value, alpha=0.25)

    plt.title(title, size=20, color='grey', y=1.1)
    labels = ('Non-Michelin', 'Michelin')
    plt.legend(labels, loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()

categories = ['food', 'service', 'price', 'decor', 'excellent_to_terrible_ratio', 'value']
normalized_data = grouped_data.copy()
for col in categories:
    max_value = normalized_data[col].max()
    normalized_data[col] = normalized_data[col] / max_value * 50 

make_radar_plot(normalized_data, 'Michelin vs Non-Michelin Restaurant Comparison', categories)
