# This file is used to visualise the michelin star restaurants 

import folium
import pandas as pd

df = pd.read_csv('../data/michelin_my_maps.csv')
def create_standard_map():
    map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Name']}<br>{row.get('Cuisine', 'Cuisine not specified')}<br>{row.get('Price', 'Price not specified')}",
            tooltip=row['Name'],
            icon=folium.Icon(color='red', icon='star')
        ).add_to(map)
    map.save('michelin_restaurants_map.html')

def create_city_map(city_name, center_coords, zoom_level=13):
    city_map = folium.Map(location=center_coords, zoom_start=zoom_level)
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(f"{row['Name']}", parse_html=True),
            tooltip=row['Name']
        ).add_to(city_map)
    city_map.save(f'{city_name}_michelin_map.html')

nyc_center_coords = [40.7128, -74.0060] 
dublin_center_coords = [53.3498, -6.2603]
create_city_map('NYC', nyc_center_coords)
create_city_map( 'Dublin', dublin_center_coords)
