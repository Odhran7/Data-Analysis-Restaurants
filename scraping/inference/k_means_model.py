from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_with_sentiment.csv')
X = df.drop('InMichelin', axis=1)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
silhouette_scores = []
K_range = range(2, 11) 
for K in K_range:
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Values of k')
plt.show()
best_K = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Best number of clusters based on silhouette score: {best_K}")

kmeans_final = KMeans(n_clusters=best_K, init='k-means++', n_init=10, max_iter=300, random_state=42)
kmeans_final.fit(X_scaled)
df['Cluster'] = kmeans_final.labels_
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans_final.cluster_centers_), columns=X.columns)
print("Cluster centers:")
print(cluster_centers)

cluster_stats = df.groupby('Cluster').agg(['mean', 'median', 'count'])
print("Cluster statistics:")
print(cluster_stats)

for feature in df.columns.drop('Cluster'):
    print(f"\nStatistics for '{feature}' feature across clusters:")
    feature_stats = df.groupby('Cluster')[feature].agg(['mean', 'median', 'count'])
    print(feature_stats)

"""
Best number of clusters based on silhouette score: 4
Cluster centers:
        food    service  ...  sentiment_polarity  sentiment_subjectivity
0  21.019048  19.266667  ...            0.321925                0.594778
1  46.222222  45.861111  ...            0.333706                0.595863
2  22.428571  20.285714  ...            0.157789                0.566650
3  37.291667  36.750000  ...            0.344458                0.600648

[4 rows x 12 columns]
Cluster statistics:
        InMichelin               ... sentiment_subjectivity             

              mean median count  ...                   mean    median count
Cluster                          ...                                    

0         0.447619    0.0   105  ...               0.594985  0.594987    60
1         0.883333    1.0   180  ...               0.596191  0.594133   145
2         0.285714    0.0     7  ...               0.562008  0.558196     6
3         0.687500    1.0    48  ...               0.601363  0.596733    43

[4 rows x 39 columns]

Statistics for 'InMichelin' feature across clusters:
             mean  median  count
Cluster
0        0.447619     0.0    105
1        0.883333     1.0    180
2        0.285714     0.0      7
3        0.687500     1.0     48

Statistics for 'food' feature across clusters:
              mean  median  count
Cluster
0        21.019048    21.0    105
1        46.222222    45.0    180
2        22.428571    20.0      7
3        37.291667    45.0     48

Statistics for 'service' feature across clusters:
              mean  median  count
Cluster
0        19.266667    19.0    105
1        45.861111    45.0    180
2        20.285714    19.0      7
3        36.750000    40.0     48

Statistics for 'price' feature across clusters:
              mean  median  count
Cluster
0        46.647619    45.0    105
1        41.388889    40.0    180
2        45.000000    45.0      7
3        51.687500    40.0     48

Statistics for 'decor' feature across clusters:
              mean  median  count
Cluster
0        18.857143    19.0    105
1        43.561644    45.0    146
2        23.428571    20.0      7
3        35.625000    40.0     48

Statistics for 'number_of_reviews' feature across clusters:
                mean  median  count
Cluster
0         763.571429   598.0    105
1         746.883333   512.0    180
2        3062.428571  2552.0      7
3        2701.062500  2425.0     48

Statistics for 'excellent' feature across clusters:
                mean  median  count
Cluster
0         272.580952   145.0    105
1         217.250000   117.0    180
2         832.285714   724.0      7
3        1363.458333  1166.5     48

Statistics for 'very_good' feature across clusters:
               mean  median  count
Cluster
0         99.266667    84.0    105
1         42.911111    23.5    180
2        626.571429   515.0      7
3        337.729167   313.5     48

Statistics for 'average' feature across clusters:
               mean  median  count
Cluster
0         33.933333    23.0    105
1         17.327778    10.0    180
2        283.285714   269.0      7
3        142.354167   143.5     48

Statistics for 'poor' feature across clusters:
               mean  median  count
Cluster
0         15.257143    11.0    105
1          8.127778     4.0    180
2        151.428571   146.0      7
3         60.687500    59.5     48

Statistics for 'terrible' feature across clusters:
               mean  median  count
Cluster
0         14.676190     9.0    105
1          6.594444     4.0    180
2        180.571429   173.0      7
3         43.979167    42.0     48

Statistics for 'sentiment_polarity' feature across clusters:
             mean    median  count
Cluster
0        0.316725  0.324706     60
1        0.334877  0.333772    145
2        0.129278  0.070793      6
3        0.346272  0.332581     43

Statistics for 'sentiment_subjectivity' feature across clusters:        
             mean    median  count
Cluster
0        0.594985  0.594987     60
1        0.596191  0.594133    145
2        0.562008  0.558196      6
3        0.601363  0.596733     43

Cluster 0 might represent average-quality restaurants that are not particularly remarkable but provide consistent service.
Cluster 1 seems to include high-performing, Michelin-quality restaurants that are recognized for their superior food and service.
Cluster 2 could be larger or more popular restaurants that don't necessarily translate popularity into quality, as seen by the lower proportion in the Michelin Guide and more mixed reviews.
Cluster 3 represents restaurants that are possibly up-and-coming, with good food and decor and positive reviews, which may be reflected in their higher Michelin presence.
"""