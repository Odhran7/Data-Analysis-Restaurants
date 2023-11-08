import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans
from read_merged_csv import read_merged


df = read_merged()
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed.drop(['InMichelin'], axis=1))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_imputed['pca-one'] = pca_result[:, 0]
df_imputed['pca-two'] = pca_result[:, 1]

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)
df_imputed['KMeans_Cluster'] = kmeans_clusters

dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(X_scaled)
df_imputed['DBSCAN_Cluster'] = dbscan_clusters

feature_names = df_imputed.columns.drop(['InMichelin', 'pca-one', 'pca-two', 'DBSCAN_Cluster', 'KMeans_Cluster'])

plt.figure(figsize=(24, 24))

small_font_size = 8
plt.rcParams.update({'font.size': small_font_size})

ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="KMeans_Cluster",
    palette=sns.color_palette("hsv", num_clusters),
    data=df_imputed,
    legend="full",
    alpha=0.7,
    ax=ax1
)
ax1.set_title('PCA Results Colored by KMeans Cluster Label')

ax2 = plt.subplot2grid((5, 2), (0, 1), colspan=1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="DBSCAN_Cluster",
    palette=sns.color_palette("hsv", len(np.unique(dbscan_clusters))),
    data=df_imputed,
    legend="full",
    alpha=0.7,
    ax=ax2
)
ax2.set_title('PCA Results Colored by DBSCAN Cluster Label')

def calculate_michelin_percentage(df, cluster_label):
    cluster_data = df[df[cluster_label] != -1] 
    michelin_percentages = cluster_data.groupby(cluster_label)['InMichelin'].mean() * 100
    return michelin_percentages

michelin_percentage_kmeans = calculate_michelin_percentage(df_imputed, 'KMeans_Cluster')
michelin_percentage_dbscan = calculate_michelin_percentage(df_imputed, 'DBSCAN_Cluster')

# Print Michelin percentages for KMeans
print("KMeans Michelin percentages:")
print(michelin_percentage_kmeans)

# Print Michelin percentages for DBSCAN
print("\nDBSCAN Michelin percentages:")
print(michelin_percentage_dbscan)

for i, cluster in enumerate(np.unique(kmeans_clusters)):
    michelin_percent = michelin_percentage_kmeans.get(cluster, 0) 
    ax = plt.subplot2grid((5, 2), (1 + i, 0))
    cluster_features = X_scaled[kmeans_clusters == cluster]
    pca = PCA(n_components=2).fit(cluster_features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    sns.heatmap(pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2']),
                annot=True, cmap='coolwarm', cbar=False, ax=ax)
    ax.set_title(f'KMeans Cluster {i} - Michelin %: {michelin_percent:.2f}%')
    print(f"\nKMeans Cluster {i} Loadings:")
    print(pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2']))


for j, cluster in enumerate(np.unique(dbscan_clusters)):
    if cluster == -1: 
        continue
    michelin_percent = michelin_percentage_dbscan.get(cluster, 0) 
    ax = plt.subplot2grid((5, 2), (1 + j, 1))
    cluster_features = X_scaled[dbscan_clusters == cluster]
    pca = PCA(n_components=2).fit(cluster_features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    sns.heatmap(pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2']),
                annot=True, cmap='coolwarm', cbar=False, ax=ax)
    ax.set_title(f'DBSCAN Cluster {cluster} - Michelin %: {michelin_percent:.2f}%')
    print(f"\nDBSCAN Cluster {cluster} Loadings:")
    print(pd.DataFrame(loadings, index=feature_names, columns=['PC1', 'PC2']))

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()