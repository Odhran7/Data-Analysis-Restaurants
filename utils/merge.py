# This file is used to merge the two datasets into one

import pandas as pd

def merge(original_df, new_df):
    merged_df = pd.merge(original_df, new_df, left_on='Restaurant Name', right_on='Query')
    merged_df.fillna(0, inplace=True)
    merged_df.rename(columns={'Restaurant Name_x': 'Original Restaurant Name', 'Restaurant Name_y': 'Scraped Restaurant Name'}, inplace=True)
    merged_df.drop(columns=['idx', 'Query'], inplace=True)
    columns_order = [
        'Original Restaurant Name',
        'Scraped Restaurant Name',
        'Address',
        'URL',
        'Restaurant URL',
        'Cuisines',
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
    ]
    merged_df = merged_df[columns_order]
    merged_df.to_csv('../output/v2/merged.csv', index=False)

original_df = pd.read_csv('../data/MichelinNY.csv', encoding='ISO-8859-1')
new_df = pd.read_csv('../output/v2/extraMichelinNYv2.csv', encoding='ISO-8859-1')

merge(original_df, new_df)