# This file is for conducting pca & cluster analysis on the original data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# The first thing we need to do is standardise the data 
numerical_columns = ['Food', 'Decor', 'Service', 'Price']
df = pd.read_csv('../data/MichelinNY.csv',  usecols=numerical_columns, encoding='ISO-8859-1')

# Normally standardise the data with mean 0 and std 1
scaler = StandardScaler()
standardised_df = df.copy()
standardised_data = pd.DataFrame(scaler.fit_transform(standardised_df))

# Now we can conduct PCA on the data
# How many components should we use?
pca = PCA(n_components=4)

pca.fit_transform(standardised_data)

explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance.cumsum()

fig, ax1 = plt.subplots(figsize=(10, 6))
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
