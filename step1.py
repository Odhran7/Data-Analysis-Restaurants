import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dotenv import load_dotenv
import os
import requests

# Loading env variables

load_dotenv(dotenv_path='.env')
api_key = os.environ.get('RESTAURANT_API_KEY')

# Reading the data 

df = pd.read_csv('data/MichelinNY.csv', encoding='ISO-8859-1')

"""
Characteristics of the data
InMichelin: A binary variable indicating whether the restaurant is included in the Michelin guide (1) or not (0).
Restaurant Name: The name of the restaurant.
Food: A rating for the food, ranging from 15 to 28.
Decor: A rating for the decor, ranging from 12 to 28.
Service: A rating for the service, ranging from 13 to 28.
Price: The price, ranging from 13 to 201.
"""

# Step 1: Describing and Visualizing the Data

# This function accepts the df as read in above and returns the distribution of the different cols grouped by whether or not the restaruants are Michelin rated

def create_distributions_by_michelin(df):
    michelin_restaurants = df[df['InMichelin'] == 1]
    non_michelin_restaurants = df[df['InMichelin'] == 0]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    sns.histplot(data=michelin_restaurants, x='Food', bins=20, kde=True, color='blue', ax=axs[0, 0], label='Michelin')
    sns.histplot(data=non_michelin_restaurants, x='Food', bins=20, kde=True, color='red', ax=axs[0, 0], label='Not Michelin')
    
    sns.histplot(data=michelin_restaurants, x='Decor', bins=20, kde=True, color='blue', ax=axs[0, 1], label='Michelin')
    sns.histplot(data=non_michelin_restaurants, x='Decor', bins=20, kde=True, color='red', ax=axs[0, 1], label='Not Michelin')
    
    sns.histplot(data=michelin_restaurants, x='Service', bins=20, kde=True, color='blue', ax=axs[1, 0], label='Michelin')
    sns.histplot(data=non_michelin_restaurants, x='Service', bins=20, kde=True, color='red', ax=axs[1, 0], label='Not Michelin')
    
    sns.histplot(data=michelin_restaurants, x='Price', bins=20, kde=True, color='blue', ax=axs[1, 1], label='Michelin')
    sns.histplot(data=non_michelin_restaurants, x='Price', bins=20, kde=True, color='red', ax=axs[1, 1], label='Not Michelin')
    
    axs[0, 0].set_title('Food')
    axs[0, 1].set_title('Decor')
    axs[1, 0].set_title('Service')
    axs[1, 1].set_title('Price')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# This function accepts the df as input and prints out some basic info about the df	

def describe_data(df):
    michelin_restaurants = df[df['InMichelin'] == 1]
    non_michelin_restaurants = df[df['InMichelin'] == 0]
    
    num_michelin = df['InMichelin'].sum()
    num_restaurants = len(df['InMichelin'])
    
    average_food_michelin = michelin_restaurants['Food'].mean()
    average_food_non_michelin = non_michelin_restaurants['Food'].mean()
    average_food_total = df['Food'].mean()
    
    average_decor_michelin = michelin_restaurants['Decor'].mean()
    average_decor_non_michelin = non_michelin_restaurants['Decor'].mean()
    average_decor_total = df['Decor'].mean()
    
    average_service_michelin = michelin_restaurants['Service'].mean()
    average_service_non_michelin = non_michelin_restaurants['Service'].mean()
    average_service_total = df['Service'].mean()
    
    average_price_michelin = michelin_restaurants['Price'].mean()
    average_price_non_michelin = non_michelin_restaurants['Price'].mean()
    average_price_total = df['Price'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=[
        ["Number of Restaurants", num_michelin, num_restaurants - num_michelin, num_restaurants],
        ["Average Food Rating", round(average_food_michelin, 2), round(average_food_non_michelin, 2), round(average_food_total, 2)],
        ["Average Decor Rating", round(average_decor_michelin, 2), round(average_decor_non_michelin, 2), round(average_decor_total, 2)],
        ["Average Service Rating", round(average_service_michelin, 2), round(average_service_non_michelin, 2), round(average_service_total, 2)],
        ["Average Price", round(average_price_michelin, 2), round(average_price_non_michelin, 2), round(average_price_total, 2)]
    ],
    colLabels=["Statistic", "Michelin Restaurants", "Non-Michelin Restaurants", "Total"],
    cellLoc='center',
    loc='center')
    plt.show()

# This function creates a network of the restaurants based on the parameter passed in and the threshold passed in

def create_network(df, param, threshold):
    G = nx.Graph()
    for index, row in df.iterrows():
        G.add_node(row['Restaurant Name'], rating=row[param], michelin=True if row['InMichelin'] == 1 else False)
    for rating in range(len(df)):
        for compare_rating in range(rating + 1, len(df)):
            if abs(df.iloc[rating][param] - df.iloc[compare_rating][param]) <= 1:
                G.add_edge(df.iloc[rating]['Restaurant Name'], df.iloc[compare_rating]['Restaurant Name'])
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, v in G.nodes(data=True) if v['michelin']],
                           node_size=50, node_color='red', label='Michelin')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, v in G.nodes(data=True) if not v['michelin']],
                           node_size=50, node_color='blue', label='Not Michelin')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.legend()
    plt.show()

def scrape_data(url):
    payload = {
        "currency": "EUR",
        "language": "en_US",
        "location_id": "15333482"
    }
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "worldwide-restaurants.p.rapidapi.com"
    }

    response = requests.post(url, data=payload, headers=headers)

    print(response.json())

scrape_data("https://worldwide-restaurants.p.rapidapi.com/detail")

