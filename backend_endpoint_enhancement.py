import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
CORS(app)

DATA_FILE = 'spending_data.csv'

# Helper function to preprocess data
def preprocess_data(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Feature: Day of the Week
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Feature: IsWeekend (1 if Saturday or Sunday, else 0)
    data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Encode 'Category' and 'Type' as numerical features for clustering
    if 'Category' in data.columns:
        data['Category'].fillna('Unknown', inplace=True)
        le_category = LabelEncoder()
        data['CategoryEncoded'] = le_category.fit_transform(data['Category'])

    if 'Type' in data.columns:
        data['Type'].fillna('Unknown', inplace=True)
        le_type = LabelEncoder()

    # Normalize 'Amount' after handling missing or non-numeric values
    data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)
    return data

# Route for clustering by spending behaviors
@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    if not os.path.exists(DATA_FILE):
        return jsonify(
            {"error": "No data available. Please upload a file first."})

    try:
        # Load and preprocess the data
        data = pd.read_csv(DATA_FILE)
        data = preprocess_data(data)

        # Selecting features for clustering
        clustering_features = ['Amount', 'DayOfWeek', 'IsWeekend',
                               'CategoryEncoded', 'TypeEncoded']
        clustering_data = data[clustering_features].fillna(0)

        # Standardize features for better clustering performance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)

        # KMeans clustering
        kmeans = KMeans(n_clusters=3,
                        random_state=42)  # Adjust n_clusters as needed
        data['Cluster'] = kmeans.fit_predict(scaled_features)

        # Save the updated data with cluster labels
        data.to_csv(DATA_FILE, index=False)

        return jsonify({
                           "message": "Clustering based on spending behaviors successfully performed"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Get Insights with enhanced clustering
@app.route('/insights', methods=['GET'])
def get_insights():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."})

    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)

        # Calculate insights for each cluster
        top_transactions = data.groupby('Cluster').apply(
            lambda x: x.nlargest(5, 'Amount')[['Date', 'Name', 'Category', 'Amount', 'Cluster']]
        ).reset_index(drop=True)

        frequent_expenses = data.groupby(['Cluster', 'Category']).size().reset_index(name='Frequency')
        top_frequent_expenses = frequent_expenses.sort_values(
            ['Cluster', 'Frequency'], ascending=[True, False]
        ).groupby('Cluster').head(3).reset_index(drop=True)

        # Outliers using IQR method
        outliers = []
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            Q1 = cluster_data['Amount'].quantile(0.25)
            Q3 = cluster_data['Amount'].quantile(0.75)
            IQR = Q3 - Q1
            high_outliers = cluster_data[cluster_data['Amount'] > (Q3 + 1.5 * IQR)]
            low_outliers = cluster_data[cluster_data['Amount'] < (Q1 - 1.5 * IQR)]
            outliers.append({
                "Cluster": int(cluster),
                "High Outliers": high_outliers[['Date', 'Name', 'Category', 'Amount']].to_dict(orient='records'),
                "Low Outliers": low_outliers[['Date', 'Name', 'Category', 'Amount']].to_dict(orient='records')
            })

        # Convert insights into JSON
        insights = {
            "top_transactions": top_transactions.to_dict(orient='records'),
            "top_frequent_expenses": top_frequent_expenses.to_dict(orient='records'),
            "outliers": outliers
        }
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)