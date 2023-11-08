import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from read_merged_csv import read_merged
import matplotlib.gridspec as gridspec

# Read the data
df = read_merged()

# Preprocessing
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed.drop(['InMichelin'], axis=1))

# Perform PCA and add the results to the dataframe
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_imputed['pca-one'] = pca_result[:, 0]
df_imputed['pca-two'] = pca_result[:, 1]

# Perform KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)
df_imputed['KMeans_Cluster'] = kmeans_clusters

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)
df_imputed['DBSCAN_Cluster'] = dbscan_clusters

# Feature names for the heatmaps
feature_names = df_imputed.columns.drop(['InMichelin', 'pca-one', 'pca-two', 'KMeans_Cluster', 'DBSCAN_Cluster'])

# Define the number of top features and PCs for the heatmap
num_top_features = 3
num_pcs = 5

def plot_pca_heatmaps(cluster_labels, X_scaled, feature_names, num_clusters, scaler, title):
    plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(num_clusters + 1, 1, height_ratios=[1] * num_clusters + [5])

    # Scatter plot for PCA results
    ax0 = plt.subplot(gs[0])
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=cluster_labels,
        palette=sns.color_palette("hsv", len(np.unique(df_imputed[cluster_labels]))),
        data=df_imputed,
        legend="full",
        alpha=0.7,
        ax=ax0
    )
    ax0.set_title(title)

    # Plot PCA heatmaps for each cluster
    for i, cluster in enumerate(np.unique(df_imputed[cluster_labels])):
        if cluster == -1:  # Skip noise if using DBSCAN
            continue

        cluster_data = df_imputed[df_imputed[cluster_labels] == cluster]
        pca = PCA(n_components=num_pcs)
        pca.fit(scaler.transform(cluster_data[feature_names]))
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

        top_indices = np.argsort(-np.abs(loadings), axis=0)[:num_top_features]
        top_loadings = loadings[top_indices, :].T

        # We reshape the top_loadings to be 2-dimensional if necessary
        heatmap_data = np.array(top_loadings)
        if heatmap_data.ndim > 2:
            heatmap_data = heatmap_data.reshape((num_pcs, -1))

        heatmap_labels = feature_names[top_indices].reshape(-1)

        # Create a DataFrame from the loadings for better labeling in the heatmap
        heatmap_df = pd.DataFrame(
            data=heatmap_data,
            columns=heatmap_labels,
            index=[f'PC{i+1}' for i in range(heatmap_data.shape[0])]
        )
        
        ax = plt.subplot(gs[i + 1])
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='coolwarm',
            cbar=True,
            center=0,
            ax=ax
        )
        ax.set_title(f'Cluster {cluster} - Top {num_top_features} PCA Loadings')
    plt.tight_layout()
    plt.show()
# Plot heatmaps for KMeans
plot_pca_heatmaps('KMeans_Cluster', X_scaled, feature_names, num_clusters, scaler, 'PCA Results Colored by KMeans Cluster Label')

# Plot heatmaps for DBSCAN
num_dbscan_clusters = len(np.unique(df_imputed['DBSCAN_Cluster'])) - (1 if -1 in df_imputed['DBSCAN_Cluster'].values else 0)
plot_pca_heatmaps('DBSCAN_Cluster', X_scaled, feature_names, num_dbscan_clusters, scaler, 'PCA Results Colored by DBSCAN Cluster Label')
