# This file is used to merge the two datasets into one

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def merge(original_df, new_df):
    # Merge the two datasets
    merged_df = pd.merge(original_df, new_df, left_on='Restaurant Name', right_on='Query')
    merged_df.fillna({'Cuisines': 'Unknown'}, inplace=True)
    merged_df.rename(columns={'Restaurant Name_x': 'Original Restaurant Name', 'Restaurant Name_y': 'Scraped Restaurant Name'}, inplace=True)
    merged_df.drop(columns=['idx', 'Query'], inplace=True)

    # Replace 0 and NaN values in Cuisines column with 'Unknown' and ensure all values are strings
    merged_df['Cuisines'] = merged_df['Cuisines'].replace(0, 'Unknown').astype(str)

    # Split combinations of cuisines into individual values
    merged_df['Cuisines'] = merged_df['Cuisines'].str.split(', ')

    # Remove "Cuisines_" prefix
    merged_df['Cuisines'] = merged_df['Cuisines'].apply(lambda x: [cuisine.replace('Cuisines_', '') for cuisine in x])

    # Quantify the cuisines
    encoder = OneHotEncoder(sparse=False)
    cuisine_encoded = encoder.fit_transform(merged_df['Cuisines'].explode().values.reshape(-1, 1))
    cuisine_encoded_df = pd.DataFrame(cuisine_encoded, columns=encoder.get_feature_names_out(['Cuisines']))
    merged_df = pd.concat([merged_df, cuisine_encoded_df], axis=1)
    merged_df.drop('Cuisines', axis=1, inplace=True)

    # Reorder the columns
    columns_order = [
        'Original Restaurant Name',
        'Scraped Restaurant Name',
        'Address',
        'URL',
        'Restaurant URL',
        'InMichelin',
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
    ] + list(encoder.get_feature_names_out(['Cuisines']))
    merged_df = merged_df[columns_order]
    merged_df = merged_df.fillna(0)

    # Save the processed data to a new CSV file
    merged_df.to_csv('../output/v2/merged.csv', index=False)

original_df = pd.read_csv('../data/MichelinNY.csv', encoding='ISO-8859-1')
new_df = pd.read_csv('../output/v2/extraMichelinNYv2.csv', encoding='ISO-8859-1')

merge(original_df, new_df)
