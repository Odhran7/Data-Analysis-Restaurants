import pandas as pd
from textblob import TextBlob

data = pd.read_csv('../output/v2/merged.csv')
reviews = pd.read_csv('../output/v2/review.csv', error_bad_lines=False, warn_bad_lines=True)
def calculate_sentiment(dataframe):
    dataframe['sentiment_polarity'] = dataframe['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
    dataframe['sentiment_subjectivity'] = dataframe['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return dataframe

def calculate_average_sentiment(reviews_dataframe):
    average_sentiment = reviews_dataframe.groupby('name').agg({
        'sentiment_polarity': 'mean',
        'sentiment_subjectivity': 'mean'
    }).reset_index()
    return average_sentiment

michelin_data = pd.read_csv('michelin_data_with_sentiment.csv')
merged_data = pd.read_csv('../output/v2/merged_with_sentiment.csv')
