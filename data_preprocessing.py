import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Step 1: Load the dataset
data = pd.read_csv("data/spending_data.csv")

# Step 2: Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Feature extraction
# Extract day of the week from the date
data['Date'] = pd.to_datetime(data['Date'])
data['Day0fWeek'] = data['Date'].dt.dayofweek

# Step 4: Select relevant feature for clustering
features = data[['Amount', 'Day0fWeek']]

# Step 5: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 6: Apply K-means clustering
# Define the number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)
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
print(f"Silhouette score: {silhouette_avg:.2f}")


