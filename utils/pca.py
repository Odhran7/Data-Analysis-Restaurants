from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from read_csv import read_original
from read_merged_csv import read_merged

def create_scree_plot(n, numerical_columns, is_merged):
    df = read_merged(numerical_columns) if is_merged else read_original(numerical_columns)

    scaler = StandardScaler()
    standardised_data = pd.DataFrame(scaler.fit_transform(df))

    pca = PCA(n_components=n)
    pca.fit_transform(standardised_data)

    explained_variance = pca.explained_variance_ratio_

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, explained_variance, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()

def interpret_pca(numerical_columns, is_merged):
    df = read_merged(numerical_columns) if is_merged else read_original(numerical_columns)
    print(df.isnull().any())
    scaler = StandardScaler()
    standardised_data = pd.DataFrame(scaler.fit_transform(df))

    pca = PCA()
    pca_data = pca.fit_transform(standardised_data)

    loadings = pca.components_.T
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    results_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(loadings.shape[1])], index=numerical_columns)
    results_df.loc['Eigenv'] = pca.explained_variance_
    results_df.loc['Propor'] = explained_variance_ratio
    results_df.loc['Cumula'] = cumulative_explained_variance

    print(results_df)
    loadings_df = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(len(pca.components_))], index=numerical_columns)

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings_df, cmap='coolwarm', annot=True)
    plt.xlabel('Principal Component')
    plt.ylabel('Variable')
    plt.title('PCA Loadings Heatmap')
    plt.show()

def interpret_pca_by_michelin(numerical_columns, michelin_column, is_merged):
    df = read_merged() if is_merged else read_original()

    scaler = StandardScaler()
    standardised_data = scaler.fit_transform(df[numerical_columns])

    pca = PCA()
    pca_data = pca.fit_transform(standardised_data)

    pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    pca_df[michelin_column] = df[michelin_column]

    logistic_regression_curves(pca_df, michelin_column)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: 'red', 1: 'blue'}
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df[michelin_column].apply(lambda x: colors[x]), alpha=0.5)
    ax.set_title('PCA Plot')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    legend_labels = {0: 'Not Michelin', 1: 'Michelin'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10) for label in colors]
    ax.legend(handles, [legend_labels[label] for label in colors], title='Michelin')

    plt.show()

def logistic_regression_curves(pca_df, michelin_column):
    y = pca_df[michelin_column]
    pcs = [col for col in pca_df.columns if col.startswith('PC')]

    n_cols = 2
    n_rows = -(-len(pcs) // n_cols) 

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 6 * n_rows))
    axes = axes.flatten()

    for i, pc in enumerate(pcs):
        X = pca_df[pc].values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(X, y)

        x_values = np.linspace(X.min(), X.max(), 300)
        y_probs = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

        sns.scatterplot(x=pc, y=michelin_column, data=pca_df, alpha=0.5, ax=axes[i])
        axes[i].plot(x_values, y_probs, color='red')
        axes[i].set_xlabel(pc)
        axes[i].set_ylabel('Probability of Being Michelin')
        axes[i].set_title(f'Logistic Regression Curve for {pc}')

    plt.tight_layout()
    plt.show()



numerical_cols_merged = [
    'Food', 
    'Decor', 
    'Service', 
    'Price', 
    'Number of Reviews', 
    'Excellent reviews', 
    'Very good reviews', 
    'Average reviews', 
    'Poor reviews', 
    'Terrible reviews'
]

numerical_cols_original = [
    'Food',
    'Decor',
    'Service',
    'Price'
]

interpret_pca_by_michelin(numerical_cols_original, "InMichelin", False)
