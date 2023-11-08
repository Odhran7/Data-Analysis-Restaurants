import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv('merged_with_sentiment.csv')
df['excellent_to_terrible_ratio'] = df['excellent'] / (df['terrible'] + 1)
df['value'] = df['food'] / (df['price'] + 1) 
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_imputed.drop(['InMichelin'], axis=1)) 
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
sorted_distances = np.sort(distances[:, 4], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(sorted_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('5th Nearest Neighbor Distance for Each Point')
plt.show()
eps_value = 2.5
dbscan = DBSCAN(eps=eps_value, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled) 
df_imputed['pca-one'] = pca_result[:,0]
df_imputed['pca-two'] = pca_result[:,1]
df_imputed['DBSCAN_Cluster'] = clusters
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="DBSCAN_Cluster",
    palette=sns.color_palette("hsv", len(df_imputed['DBSCAN_Cluster'].unique())),
    data=df_imputed,
    legend="full",
    alpha=0.3
)
plt.title('PCA Results Colored by DBSCAN Cluster Label')
plt.show()

loadings = pca.components_.T

# Plot heatmap of loadings
plt.figure(figsize=(10, 7))
sns.heatmap(loadings, annot=True, cmap='viridis',
            yticklabels=df.columns.drop('InMichelin'),
            xticklabels=[f"PC{i+1}" for i in range(loadings.shape[1])])
plt.title("Heatmap of PCA Loadings")
plt.show()




