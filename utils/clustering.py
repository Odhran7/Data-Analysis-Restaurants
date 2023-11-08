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
def analyze_clusters_v4(n_clusters, data, numerical_cols, linkage, affinity):
    # Standardize the numerical columns
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[numerical_cols])

    # Perform clustering
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
    linkage_methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    distance_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    n_clusters_range = range(2, 11)
    highest_score = -np.inf
    best_combination = None
    lambda_factor = 0.5
    for n_clusters in n_clusters_range:
        for linkage_method in linkage_methods:
            for distance_metric in distance_metrics:
                if linkage_method == 'ward' and distance_metric != 'euclidean':
                    continue

                try:
                    avg_values_clusters, michelin_proportion, avg_michelin_proportion, cluster_sizes = analyze_clusters_v4(
                        n_clusters, data, numerical_cols, linkage_method, distance_metric
                    )
                    cluster_entropy = entropy(cluster_sizes / np.sum(cluster_sizes))
                    std_dev_penalty = np.std(cluster_sizes)
                    score = 1 / (cluster_entropy + lambda_factor * std_dev_penalty)
                    if score > highest_score:
                        highest_score = score
                        best_combination = (n_clusters, linkage_method, distance_metric)
                except (ValueError, TypeError) as e:
                    print(f"Error for n_clusters={n_clusters}, linkage_method={linkage_method}, distance_metric={distance_metric}: {e}")

    if best_combination is None:
        print('No valid combination found.')
    else:
        print(f'Best combination:')
        print(f'Number of clusters: {best_combination[0]}')
        print(f'Linkage method: {best_combination[1]}')
        print(f'Distance metric: {best_combination[2]}')
        print(f'Highest score: {highest_score}')
# find_best_combination(data, numerical_cols)
"""
Best combination:
Number of clusters: 3
Linkage method: complete
Distance metric: correlation
Highest score: 0.25067221172118465
"""
def create_optimal_dendrogram_with_clusters(data, numerical_cols, n_clusters):
    numerical_data = data[numerical_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(numerical_data)
    linkage_method = 'complete'
    distance_metric = 'correlation'
    Z = linkage(standardized_data, method=linkage_method, metric=distance_metric)
    plt.figure(figsize=(25, 10))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=10)
    plt.title(f'Linkage: {linkage_method}\nDistance: {distance_metric}')
    plt.axhline(y=Z[-(n_clusters-1), 2], color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()

# create_optimal_dendrogram_with_clusters(data, numerical_cols, 3)
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
    print("Average values for each cluster:\n", avg_values_clusters)
    cluster_counts = clustered_data['Cluster'].value_counts()
    print("\nNumber of restaurants in each cluster:\n", cluster_counts)
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    print("\nProportion of Michelin-starred restaurants in each cluster:\n", michelin_proportion)
    interpret_with_michelin_proportion_and_table(clustered_data, numerical_cols)

    return michelin_proportion

# This function highlights the clusters with high michelin proportion
def interpret_with_michelin_proportion(clustered_data, numerical_cols):
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    high_michelin_clusters = michelin_proportion[michelin_proportion > 0.5].index.tolist()
    pairplot = sns.pairplot(data=clustered_data, hue='Cluster', vars=numerical_cols, palette='bright', plot_kws={'alpha': 0.5})
    handles, labels = pairplot.axes[0][0].get_legend_handles_labels()
    new_handles = []
    for handle, label in zip(handles, labels):
        cluster_num = int(label)
        if cluster_num in high_michelin_clusters:
            new_handle = plt.Line2D([], [], markerfacecolor=handle.get_facecolor(), markeredgecolor='black', marker='o', markersize=10, linestyle='None')
            new_handles.append(new_handle)
        else:
            new_handles.append(handle)
    
    plt.legend(new_handles, labels, title='Pair P', loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.show()

    return michelin_proportion

def interpret_with_michelin_proportion_and_table(clustered_data, numerical_cols):
    michelin_proportion = clustered_data.groupby('Cluster')['InMichelin'].mean()
    pairplot = sns.pairplot(data=clustered_data, hue='Cluster', vars=numerical_cols, palette='bright', plot_kws={'alpha': 0.5})
    plt.subplots_adjust(right=0.8)
    cell_text = [[f"{prop:.2f}"] for prop in michelin_proportion]
    row_labels = [f"Cluster {i}" for i in michelin_proportion.index.tolist()]
    pairplot.fig.subplots_adjust(right=0.6) 
    table_ax = pairplot.fig.add_axes([0.7, 0.1, 0.2, 0.8], frame_on=False) 
    table_ax.axis('off')
    table_ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Michelin Proportion'], cellLoc = 'center', loc='center right')
    pairplot.fig.suptitle('Pairplot of Numerical Variables with Clusters and Michelin Proportions', y=1.02)

    plt.show()
print(perform_clustering_analysis(n_clusters=3, linkage_method='complete', distance_metric='correlation', data=data, numerical_cols=numerical_cols))
