import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Step 1: Load the dataset
data = pd.read_csv("data/spending_data.csv")

# Step 2: Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Feature extraction
# Extract day of the week from the date
data['Date'] = pd.to_datetime(data['Date'])
data['Day0fWeek'] = data['Date'].dt.dayofweek

# Add a feature to indicate if the day is a weekend
data['IsWeekend'] = data['Day0fWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Convert the Category column to numeric labels using LabelEncoder
le = LabelEncoder()
data['CategoryEncoded'] = le.fit_transform(data['Category'])

# Step 4: Select relevant feature for clustering
features = data[['Amount', 'Day0fWeek', 'IsWeekend', 'CategoryEncoded']]

# Step 5: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 6: Apply K-means clustering with different number of clusters
best_score = -1
best_k = 0
for n_clusters in range(2, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    print(f'Number of Cluster: {n_clusters}, Silhouette Score: {score: .2f}')
    if score > best_score:
        best_score = score
        best_k = n_clusters

# Apply K-means with the best number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(scaled_features)

# Step 7: Add cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Step 8: Visualize the clusters
plt.scatter(data['Amount'], data['Day0fWeek'],
c=data['Cluster'], cmap='viridis')
plt.xlabel('Amount')
plt.ylabel('DayOfWeek')
plt.title('Spending Clusters')
plt.show()

# Step 9: Evaluate the clustering performance
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"Best number of Clusters: {best_k}, Silhouette score: {silhouette_avg:.2f}")

# Step 10: Calculate Davies-Bouldin Score for the further evaluation
davies_bouldin_avg = davies_bouldin_score(scaled_features, kmeans.labels_)
print(f'Davies-Bouldin Score: {davies_bouldin_avg:2f}')
