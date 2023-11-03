import pandas as pd
from scrape_michelin import scrape_location_data

df = pd.read_csv('../data/michelin_my_maps.csv')
michelin_restaurants = df['Name']

for restaurant in michelin_restaurants:
    try:
        scrape_location_data(restaurant, True)
    except Exception as e:
        print(e)
        continue