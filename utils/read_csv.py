import pandas as pd

def read_original(cols = None):
    df = None
    if cols:
        df = pd.read_csv('../data/MichelinNY.csv',  usecols=cols, encoding='ISO-8859-1')
    else:
         df = pd.read_csv('../data/MichelinNY.csv', encoding='ISO-8859-1')
    return df
