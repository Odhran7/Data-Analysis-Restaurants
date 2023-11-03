# This file will be used to conduct clustering anlysis on both the datasets
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import entropy

# Load the data
file_path = '../data/MichelinNY.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
numerical_cols = ['Food', 'Decor', 'Service', 'Price']

# Creates the dendrogram plots for all linkages and distance metrics
def create_hierarchial_clusters(data, numerical_cols):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    fig, axes = plt.subplots(nrows=len(linkage_methods), ncols=len(distance_metrics), figsize=(15, 10))
    for i, linkage_method in enumerate(linkage_methods):
        for j, distance_metric in enumerate(distance_metrics):
            if linkage_method == 'ward' and distance_metric != 'euclidean':
                continue
            Z = linkage(standardized_data, method=linkage_method, metric=distance_metric)

            dendrogram(Z, ax=axes[i, j], leaf_rotation=90, leaf_font_size=10)
            axes[i, j].set_title(f'Linkage: {linkage_method}\nDistance: {distance_metric}')
    plt.tight_layout()
    plt.show()
# create_hierarchial_clusters(data, numerical_cols)

# Creates the dendrogram plots for the clusters of n_clusters
def create_nhierarchial_clusters(n_clusters, data, numerical_cols):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    fig, axs = plt.subplots(1, len(linkage_methods), figsize=(20, 5))
    fig.suptitle(f'Dendrogram plots for {n_clusters} clusters')
    for i, linkage_method in enumerate(linkage_methods):
        Z = linkage(standardized_data, method=linkage_method, metric=distance_metrics[0])
        dendrogram(Z, ax=axs[i], leaf_rotation=90, leaf_font_size=10)
        axs[i].set_title(f'{linkage_method} linkage, {distance_metrics[0]} distance')
        threshold = Z[-(n_clusters-1), 2]
        axs[i].axhline(y=threshold, color='r', linestyle='--')
    plt.show()

# create_nhierarchial_clusters(4, data, numerical_cols)

# Helper function to analyze clusters
def analyze_clusters(n_clusters, data, numerical_cols, linkage, affinity):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    labels = clustering.fit_predict(standardized_data)
    clustered_data = data.copy()
    clustered_data['Cluster'] = labels
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    high_michelin_clusters = michelin_proportion[michelin_proportion > 0.6].index
    avg_values_4_clusters = clustered_data[clustered_data['Cluster'].isin(high_michelin_clusters)].groupby('Cluster')[['Food', 'Decor', 'Service', 'Price']].mean()
    print(avg_values_4_clusters)

# Prints the average values for each cluster
def analyze_link_affinity(n_clusters, data, numerical_cols):
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    for linkage_method in linkage_methods:
        for distance_metric in distance_metrics:
            if linkage_method == 'ward' and distance_metric != 'euclidean':
                continue
            print(f'Linkage: {linkage_method}, Distance: {distance_metric}')
            analyze_clusters(n_clusters, data, numerical_cols, linkage_method, distance_metric) 
# analyze_link_affinity(4, data, numerical_cols)

# helper function to analyze clusters
def analyze_clusters_v3(n_clusters, data, numerical_cols, linkage, affinity):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    labels = clustering.fit_predict(standardized_data)
    clustered_data = data.copy()
    clustered_data['Cluster'] = labels
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    high_michelin_clusters = michelin_proportion[michelin_proportion > 0.6].index
    avg_values_clusters = clustered_data[clustered_data['Cluster'].isin(high_michelin_clusters)].groupby('Cluster')[numerical_cols].mean()
    
    return avg_values_clusters

# Shows the average values for each cluster
def analyze_link_affinity_v2(n_clusters, data, numerical_cols):
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    avg_values_list = []
    labels = []
    for linkage_method in linkage_methods:
        for distance_metric in distance_metrics:
            if linkage_method == 'ward' and distance_metric != 'euclidean':
                continue
            avg_values_clusters = analyze_clusters_v3(n_clusters, data, numerical_cols, linkage_method, distance_metric)
            avg_values_list.append(avg_values_clusters)
            labels.append(f'{linkage_method}, {distance_metric}')
    fig, axs = plt.subplots(len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols)))
    for i, col in enumerate(numerical_cols):
        for j, avg_values_clusters in enumerate(avg_values_list):
            avg_values_clusters[col].plot(kind='bar', ax=axs[i], position=j, label=labels[j])
        axs[i].set_title(col)
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()

# analyze_link_affinity_v2(4, data, numerical_cols)
def analyze_clusters_v4(n_clusters, standardized_data, numerical_cols, linkage, affinity):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    labels = clustering.fit_predict(standardized_data)
    clustered_data = data.copy()
    clustered_data['Cluster'] = labels
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    avg_michelin_proportion = michelin_proportion.mean() 
    high_michelin_clusters = michelin_proportion[michelin_proportion > 0.6].index
    avg_values_clusters = clustered_data[clustered_data['Cluster'].isin(high_michelin_clusters)].groupby('Cluster')[numerical_cols].mean()
    cluster_sizes = np.bincount(labels)
    return avg_values_clusters, michelin_proportion, avg_michelin_proportion, cluster_sizes


def find_best_combination(data, numerical_cols):
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    n_clusters_range = range(2, 11)
    lowest_entropy = np.inf
    best_combination = None
    
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    
    for n_clusters in n_clusters_range:
        for linkage_method in linkage_methods:
            for distance_metric in distance_metrics:
                if linkage_method == 'ward' and distance_metric != 'euclidean':
                    continue
                
                try:
                    avg_values_clusters, michelin_proportion, avg_michelin_proportion, cluster_sizes = analyze_clusters_v4(n_clusters, standardized_data, numerical_cols, linkage_method, distance_metric)
                    cluster_entropy = entropy(cluster_sizes / np.sum(cluster_sizes))
                    
                    if cluster_entropy < lowest_entropy:
                        lowest_entropy = cluster_entropy
                        best_combination = (n_clusters, linkage_method, distance_metric)
                except (ValueError, TypeError) as e:
                    print(f"Error for n_clusters={n_clusters}, linkage_method={linkage_method}, distance_metric={distance_metric}")
                    print(e)
                    continue
                
    if best_combination is None:
        print('No valid combination found.')
    else:
        print(f'Best combination:')
        print(f'Number of clusters: {best_combination[0]}')
        print(f'Linkage method: {best_combination[1]}')
        print(f'Distance metric: {best_combination[2]}')
        print(f'Lowest entropy: {lowest_entropy}')
    linkage_methods = ['single', 'complete', 'average', 'ward']
    distance_metrics = ['euclidean', 'cityblock', 'chebyshev', 'minkowski']
    n_clusters_range = range(2, 11)
    lowest_entropy = np.inf
    best_combination = None
    
    for n_clusters in n_clusters_range:
        for linkage_method in linkage_methods:
            for distance_metric in distance_metrics:
                if linkage_method == 'ward' and distance_metric != 'euclidean':
                    continue
                
                try:
                    avg_values_clusters, michelin_proportion, avg_michelin_proportion, cluster_sizes = analyze_clusters_v4(n_clusters, data, numerical_cols, linkage_method, distance_metric)
                    cluster_entropy = entropy(cluster_sizes / np.sum(cluster_sizes))
                    
                    if cluster_entropy < lowest_entropy:
                        lowest_entropy = cluster_entropy
                        best_combination = (n_clusters, linkage_method, distance_metric)
                except (ValueError, TypeError) as e:
                    print(f"Error for n_clusters={n_clusters}, linkage_method={linkage_method}, distance_metric={distance_metric}")
                    print(e)
                    continue
                
    if best_combination is None:
        print('No valid combination found.')
    else:
        print(f'Best combination:')
        print(f'Number of clusters: {best_combination[0]}')
        print(f'Linkage method: {best_combination[1]}')
        print(f'Distance metric: {best_combination[2]}')
        print(f'Lowest entropy: {lowest_entropy}')

find_best_combination(data, numerical_cols)

"""
Best combination:
Number of clusters: 6
Linkage method: single
Distance metric: braycurtis
Highest score: 2.740041928721174
"""

def create_optimal_dendrogram(data, numerical_cols):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    linkage_method = 'single'
    distance_metric = 'euclidean'
    Z = linkage(standardized_data, method=linkage_method, metric=distance_metric)
    dendrogram(Z, leaf_rotation=90, leaf_font_size=10)
    plt.title(f'Linkage: {linkage_method}\nDistance: {distance_metric}')
    plt.tight_layout()
    plt.show()

# create_optimal_dendrogram(data, numerical_cols)

# Print the pair plot
def interpret(clustered_data, numerical_cols):
    pairplot = sns.pairplot(data=clustered_data, hue='Cluster', vars=numerical_cols, palette='bright', plot_kws={'alpha': 0.5})
    pairplot.fig.suptitle('Pairplot of Numerical Variables with Clusters', y=1.02)
    plt.show()



# Function used to perform clustering analysis on explicit cluster types
def perform_clustering_analysis(n_clusters, linkage_method, distance_metric, data, numerical_cols):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, affinity=distance_metric)
    labels = clustering.fit_predict(standardized_data)
    clustered_data = data.copy()
    clustered_data['Cluster'] = labels
    avg_values_clusters = clustered_data.groupby('Cluster')[numerical_cols].mean()
    print(avg_values_clusters)
    cluster_counts = clustered_data['Cluster'].value_counts()
    print(cluster_counts)
    interpret_with_michelin_proportion(clustered_data, numerical_cols)

# This function highlights the clusters with high michelin proportion
def interpret_with_michelin_proportion(clustered_data, numerical_cols):
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    high_michelin_clusters = michelin_proportion[michelin_proportion > 0.6].index
    pairplot = sns.pairplot(data=clustered_data, hue='Cluster', vars=numerical_cols, palette='bright', plot_kws={'alpha': 0.5})
    for i, (label, handle) in enumerate(zip(pairplot._legend.get_texts(), pairplot._legend.legendHandles)):
        cluster_num = int(label.get_text())
        if cluster_num in high_michelin_clusters:
            handle.set_color('black')
            handle.set_linewidth(1)
    pairplot.fig.suptitle('Pairplot of Numerical Variables with Clusters', y=1.02)
    plt.show()

    return michelin_proportion

perform_clustering_analysis(n_clusters=2, linkage_method='single', distance_metric='euclidean', data=data, numerical_cols=numerical_cols)