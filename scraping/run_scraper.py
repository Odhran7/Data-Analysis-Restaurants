# This is the main file for running the scrape 

# Please note - Output logs from this script are saved in the 'logs' folder

# Imports
import pandas as pd
from tripadvisor_scraper import scrape_location_data
from multiprocessing import Pool

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

idx = df.index

input_array = zip(idx, restaurants, isMichelin)

failed_restaurants = []

# for idx, restaurant, michelin_value in input_array:
#     try:
#         scrape_location_data(restaurant, michelin_value, idx)
#     except Exception as e:
#         print(f"Error scraping data for restaurant '{restaurant}': {str(e)}")
#         failed_restaurants.append(restaurant)

# Parallelize the scraping
def scrape_and_handle_errors(args):
    idx, restaurant, michelin_value = args
    try:
        scrape_location_data(restaurant, michelin_value, idx)
    except Exception as e:
        print(f"Error scraping data for restaurant '{restaurant}': {str(e)}")
        failed_restaurants.append(restaurant)

if __name__ == '__main__':
    with Pool(3) as p:
        p.map(scrape_and_handle_errors, input_array)
        
# Debugging stats
# print("List of failed restaurants:")
# print(len(failed_restaurants))
# print(failed_restaurants)