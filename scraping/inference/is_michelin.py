# This file is used to conduct an analysis to see if the restaurant is michelin star or not

import pandas as pd
from obtain_data import scrape_location_data
import argparse
from joblib import load
import numpy as np
import matplotlib.pyplot as plt 
import csv

# pipeline = load('../models/trained_pipeline.joblib')
# imputer = load('../models/imputer.joblib')

# parser = argparse.ArgumentParser(description='Scrape data to determine Michelin star status.')
# parser.add_argument('query', help='The query string to use for scraping Michelin data')
# args = parser.parse_args()
# query = args.query

# csv_file = 'michelin_star_probabilities.csv'
# csv_columns = ["InMichelin", "name", "food", "service", "price", "decor", "number_of_reviews", "excellent", "very_good", "average", "poor", "terrible", "sentiment_polarity", "sentiment_subjectivity"]


# restaurant_names = [
#     "Sunset Grill", "Olive Branch", "The Orchard", "Fusion House",
#     "The Spice Route", "Cinnamon Bazaar", "Jade Palace", "The Hungry Bear",
#     "The Backyard Grill", "The Corner Cafe", "Red Lantern", "The Greenhouse",
#     "The Local Eatery", "The Roasted Bean", "The Rusty Spoon", "Urban Diner",
#     "The Seafood Market", "Mountain View", "The Urban Garden", "Mama's Kitchen",
#     "The Pancake House", "Savory Bites", "The Clay Oven", "The Maple Tree",
#     "The Royal Table", "Street Food Deli", "The Breakfast Nook", "The Pizza Joint",
#     "The Steakhouse", "Bamboo Garden", "The Cupcake Bakery", "The River Café",
#     "The Curry House", "The Grilled Cheese Factory", "The Oyster Bar", "The Noodle House",
#     "The Chocolate Room", "The Cozy Corner", "The Lemon Leaf", "The Wine Cellar",
#     "The Pie Shop", "The Fisherman's Dock", "The Taco Stand", "The Salad Bar",
#     "The Chicken Coop", "The Coffee Mill", "The Ice Cream Parlor", "The Burger Shack",
#     "The Pastry Shop", "The Sushi Belt", "The Vegan Bistro", "The Waffle Window",
#     "The Pita Pit", "The Lobster Trap", "The Pasta Factory"
# ]
try:
    # for query in restaurant_names:
    values = scrape_location_data("carbone")
    print(values)
    # values = [np.nan if x is None else x for x in values]
    # scraped_data = np.array(values, dtype=float).reshape(1, -1)
    # imputed_data = imputer.transform(scraped_data)
    # probabilities  = pipeline.predict_proba(imputed_data)
    # prob_no_star, prob_star = probabilities[0]
    
    # print(f"Probability of no Michelin star: {prob_no_star:.2f}")
    # print(f"Probability of Michelin star: {prob_star:.2f}")

        # with open(csv_file, 'a', newline='') as csvfile:  # Open the file in append mode
        #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        #     csvfile.seek(0, 2) 
        #     if csvfile.tell() == 0:
        #         writer.writeheader()
        #     writer.writerow({"InMichelin": values[0], "name": values[1], "food": values[2], "service": values[3], "price": values[4], "decor": values[5], "number_of_reviews": values[6], "excellent": values[7], "very_good": values[8], "average": values[9], "poor": values[10], "terrible": values[11], "sentiment_polarity": values[12], "sentiment_subjectivity": values[13]})


    # Plot the probabilities
    
    # plt.bar(['No Michelin Star', 'Michelin Star'], probabilities[0], color=['red', 'green'])
    # plt.ylabel('Probability')
    # plt.title('Probability of Michelin Star Status')
    # plt.ylim(0, 1)
    # plt.show()
except Exception as e:
    print(e)
    raise e

