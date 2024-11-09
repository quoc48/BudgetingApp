import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Path to your dataset
DATA_FILE = 'spending_data.csv'

def preprocess_data(data):
    """Preprocess the dataset for clustering."""
    data['Date'] = pd.to_datetime(data['Date'])

    # Temporal Features
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Fill Missing Values
    data['Category'] = data['Category'].fillna('Unknown')  # Replace missing categories
    data['Type'] = data['Type'].fillna('Unknown')  # Replace missing types
    data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)  # Convert Amount to numeric, fill NaNs with 0

    # Encode Categorical Features
    le_category = LabelEncoder()
    le_type = LabelEncoder()
    data['CategoryEncoded'] = le_category.fit_transform(data['Category'])
    data['TypeEncoded'] = le_type.fit_transform(data['Type'])

    # Return relevant features for clustering
    features = ['Amount', 'DayOfWeek', 'IsWeekend', 'CategoryEncoded', 'TypeEncoded']
    return data[features].fillna(0)  # Ensure no NaN values remain

def determine_optimal_clusters(data, max_clusters=10):
    """Find the optimal number of clusters using inertia and silhouette scores."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    return inertias, silhouette_scores

def visualize_cluster_optimization(inertias, silhouette_scores, max_clusters=10):
    """Visualize the elbow method and silhouette scores."""
    x_range = range(2, max_clusters)

    # Plot inertia (elbow method)
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(x_range, inertias, marker='o')
    plt.title('Elbow Method (Inertia)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.xticks(x_range)

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(x_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(x_range)

    plt.tight_layout()
    plt.show()

def main():
    """Main function to preprocess data, optimize clusters, and visualize."""
    try:
        # Load the dataset
        data = pd.read_csv(DATA_FILE)

        # Preprocess the dataset
        processed_data = preprocess_data(data)

        # Determine the optimal number of clusters
        max_clusters = 10
        inertias, silhouette_scores = determine_optimal_clusters(processed_data, max_clusters)

        # Visualize the results
        visualize_cluster_optimization(inertias, silhouette_scores, max_clusters)

        # Print recommendations
        best_cluster_index = np.argmax(silhouette_scores) + 2  # Add 2 since range starts at 2
        print(f"Optimal number of clusters based on silhouette score: {best_cluster_index}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()