import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    'PC1': ['Volume and diversity of reviews', 'average', 'poor', 'terrible', 'very_good', 'number_of_reviews'],
    'PC2': ['Perceived quality in food, décor, and service', 'food', 'decor', 'service', np.nan, np.nan],
    'PC3': ['Sentiment extremity and subjectivity', 'sentiment_polarity', 'sentiment_subjectivity', np.nan, np.nan, np.nan],
    'PC4': ['Emphasis on price', 'price', np.nan, np.nan, np.nan, np.nan],
}

interpretation_df = pd.DataFrame(data)
interpretation_df.fillna('None', inplace=True)
unique_vals = pd.unique(interpretation_df.values.ravel())
val_to_num = {val: num for num, val in enumerate(unique_vals)}
numeric_df = interpretation_df.replace(val_to_num)
plt.figure(figsize=(10, 8))
ax = sns.heatmap(numeric_df, annot=interpretation_df, fmt='', cmap='coolwarm', cbar=False)
plt.title('Interpretation of Principal Components for Restaurant Reviews')
plt.xlabel('Principal Components')
plt.ylabel('Interpretations / Variables')
plt.show()
