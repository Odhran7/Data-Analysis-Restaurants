import pandas as pd

def read_merged(cols = None):
    df = None
    if cols:
        df = pd.read_csv('../output/v2/merged.csv',  usecols=cols, encoding='ISO-8859-1')
    else:
         df = pd.read_csv('../data/../output/v2/merged.csv', encoding='ISO-8859-1')
    return df
