import matplotlib.pyplot as plt
import pandas as pd

vals = [
    [
        "Cluster",
        "Variables", 
        "Interpretation", 
        "Michelin Percentage"
    ],
    [
        "KMeans Cluster 0",
        "high 'number_of_reviews', 'excellent', 'sentiment_polarity', and 'sentiment_subjectivity'", 
        "Very popular, positive reviews", 
        "High Michelin percentage (71.79%)", 
    ],
    [
        "KMeans Cluster 1",
        "very high 'price' and 'terrible'", 
        "Poor quality expensive restaurants", 
        "Low Michelin percentage (59.09%)", 
    ],
    [
        "KMeans Cluster 2",
        "moderate values in various aspects", 
        "Average Restaurants", 
        "Moderate Michelin percentage (44.76%)", 
    ],
    [
        "KMeans Cluster 3",
        "moderate values for most features", 
        "Good all round restaurants",
        "Highest Michelin percentage (87.93%)",
    ],
    [
        "DBSCAN Cluster 0",
        "high 'number_of_reviews', 'very_good', 'average', 'poor', low 'sentiment_polarity', and low 'sentiment_subjectivity'", 
        "Restaurants that are popular but have mixed reviews - represent inaccurate reviewing?",
        "Moderate Michelin percentage (46.23%)",
    ],
    [
        "DBSCAN Cluster 1",
        "high 'number_of_reviews','excellent' 'very_good', 'sentiment_polarity','sentiment_subjectivity','excellent_to_terrible', poor 'value'", 
        "Very positive, strong reviews, but poor value for money",
        "Highest Michelin percentage (86.19%)",
    ],
]

fig, ax = plt.subplots(figsize=(12, 6))  

ax.axis('off')

table = ax.table(cellText=vals, loc='center', cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)  # You can adjust this value as needed
table.auto_set_column_width(col=list(range(len(vals[0]))))  # Adjust columns to content

# Adjust layout to make room for the table:
plt.tight_layout()

# Save the figure as an image   
plt.savefig('table.png', bbox_inches='tight', dpi=300)

# Show the figure
plt.show()