import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


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

# Step 6: Determine explained variance to decide number of components
pca = PCA().fit(scaled_features)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1),
         explained_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs Number of Components')
plt.grid()
plt.show()

# Step 7: Reduce dimensionality using PCA for visualization purposes
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# Step 8: Apply K-means clustering with different number of clusters
best_score = -1
best_k = 0
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_scores.append(score)
    print(f'Number of Cluster: {n_clusters}, Silhouette Score: {score: .2f}')
    if score > best_score:
        best_score = score
        best_k = n_clusters

# Plot Silhouette Score to determine the best number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid()
plt.show()

# Step 9: Apply K-means with the best number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(scaled_features)

# Step 10: Reduce dimensionality using PCA for visualization purposes
data['Cluster'] = kmeans.labels_

# Step 11: Visualize the clusters using PCA-reduced features
plt.figure(figsize=(10, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1],
c=data['Cluster'], cmap='viridis', alpha=0.6, edgecolor='k')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('Spending Clusters (PCA Projection)')
plt.grid()
plt.show()

# Step 12: Evaluate the clustering performance
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print(f"Best number of Clusters: {best_k}, Silhouette score: {silhouette_avg:.2f}")

# Step 13: Calculate Davies-Bouldin Score for the further evaluation
davies_bouldin_avg = davies_bouldin_score(scaled_features, kmeans.labels_)
print(f'Davies-Bouldin Score: {davies_bouldin_avg:2f}')

# Step 14: Calculate Calinski-harabasz Score for evaluation
calinski_harabasz_avg = calinski_harabasz_score(scaled_features, kmeans.labels_)
print(f'Calinski_Harabasz Score: {calinski_harabasz_avg:.2f}')

# Step 15: Save the labeled data to a new CSV file
# data.to_csv('data/spending_data_with_clusters.csv', index=False)
# print("Labeled data saved to 'spending_data_with_cluster.csv'")

# Step 16: Analyze and visualize cluster characteristics using Seaborn
plt.figure(figsize=(12, 8))
sns.boxplot(x='Cluster', y='Amount', data=data)
plt.title('Amount Spent by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Amount Spent')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='Cluster', hue='IsWeekend', data=data)
plt.title('Cluster Distribution by Weekend Spending')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid()
plt.show()

# Step 17: Identify cluster characteristics
cluster_summary = data.groupby('Cluster').agg(
        Average_Amount=('Amount', 'mean'),
        Median_Amount=('Amount', 'median'),
        Count=('Amount', 'size'),
        Weekend_Spend_Percentage=('IsWeekend', 'mean')).reset_index()

cluster_summary['Weekend_Spend_Percentage'] *= 100
print("Cluster Characteristic Summary:")
print(cluster_summary)

# Step 18: Visualize cluster summary characteristics
plt.figure(figsize=(12, 8))
sns.barplot(x='Cluster', y='Average_Amount', data=cluster_summary)
plt.title('Average Amount Spent by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Amount Spent')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='Cluster', y='Weekend_Spend_Percentage', data=cluster_summary)
plt.title('Percentage of Weekend Spending by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Weekend Spending Percentage (%)')
plt.grid()
plt.show()