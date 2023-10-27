# This is the main file for running scripts

# Please note - Output logs from this script are saved in the 'logs' folder

# Imports
import pandas as pd
from tripadvisor_scraper import scrape_location_data

# Run the scraper
df = pd.read_csv('data/MichelinNY.csv', encoding='ISO-8859-1')
restaurants = df['Restaurant Name'].tolist()

# Obtain the list of Michelin values
isMichelin = df['InMichelin'].tolist()
def isMichelinFunc(x):
    if x == 1:
        return True
    else:
        return False
isMichelin = list(map(isMichelinFunc, isMichelin))

input_array = zip(restaurants, isMichelin)

failed_restaurants = []

for restaurant, michelin_value in input_array:
    try:
        scrape_location_data(restaurant, michelin_value)
    except Exception as e:
        print(f"Error scraping data for restaurant '{restaurant}': {str(e)}")
        failed_restaurants.append(restaurant)