import pandas as pd

def read_merged(cols = None):
    df = None
    if cols:
        df = pd.read_csv('../scraping/inference/merged_with_sentiment.csv',  usecols=cols, encoding='ISO-8859-1')
    else:
         df = pd.read_csv('../scraping/inference/merged_with_sentiment.csv', encoding='ISO-8859-1')
    df['excellent_to_terrible_ratio'] = df['excellent'] / (df['terrible'] + 1)
    df['value'] = df['food'] / (df['price'] + 1)
    return df
