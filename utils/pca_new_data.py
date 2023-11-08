import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from read_merged_csv import read_merged
from sklearn.impute import SimpleImputer
df = read_merged()

features = [
    'food', 
    'decor', 
    'service', 
    'price', 
    'number_of_reviews', 
    'excellent', 
    'very_good', 
    'average', 
    'poor', 
    'terrible',
    "sentiment_polarity",
    "sentiment_subjectivity",
    "excellent_to_terrible_ratio",
    "value"
]

imputer = SimpleImputer(strategy='mean') 
df_imputed = imputer.fit_transform(df[features])
df[features] = df_imputed

X = StandardScaler().fit_transform(df[features])

pca = PCA()
principalComponents = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(len(features))], index=features)


plt.figure(figsize=(12, 6))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm')
plt.title('PCA Loadings Heatmap')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()

pca_df = pd.DataFrame(data=principalComponents[:, :2], columns=['PC1', 'PC2'])
final_df = pd.concat([pca_df, df[['InMichelin']]], axis=1)
plt.figure(figsize=(8,8))
sns.scatterplot(x='PC1', y='PC2', hue='InMichelin', palette=['blue', 'red'], data=final_df)
plt.title('2 component PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


pc_interpretations = [
    "PC1 - Review Volume",
    "PC2 - Perceived Quality & Ambiance",
    "PC3 - Positive Sentiment",
    "PC4 - Opinionated Reviews",
    "PC5 - Expensive but Good"
]

# absolute_loadings = np.abs(loadings)
top_features_indices = []
top_features_loadings = []

for i in range(5):
    indices = np.argsort(np.abs(loadings[:, i]))[::-1][:5]
    top_features_indices.append(indices)
    top_features_loadings.append(loadings[indices, i])

heatmap_data = np.array(top_features_loadings).T
heatmap_data_df = pd.DataFrame(heatmap_data, index=np.arange(1, 6), columns=pc_interpretations)
formatted_heatmap_labels = np.empty_like(heatmap_data, dtype=object)

for i, pc_loadings in enumerate(top_features_loadings):
    for j, loading in enumerate(pc_loadings):
        feature_index = top_features_indices[i][j]
        formatted_heatmap_labels[j, i] = f"{features[feature_index]} ({loading:.2f})"

vmin = -1 if heatmap_data.min() < 0 else 0 
vmax = 1

plt.figure(figsize=(14, 5))
sns.heatmap(heatmap_data_df, annot=formatted_heatmap_labels, fmt="", cmap='coolwarm', cbar=True, vmin=vmin, vmax=vmax, center=0)
plt.title('PCA Loadings Heatmap with Interpretations')
plt.ylabel('Top 5 Features')
plt.xlabel('Principal Components and Interpretations')
plt.xticks(rotation=45)  
plt.savefig('pca_loadings_heatmap.png', bbox_inches='tight')
plt.show()