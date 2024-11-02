import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

DATA_FILE = 'spending_data.csv'


# Upload Spending Data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        new_data = pd.read_csv(file)
        # Keep only necessary columns: 'Date', 'Name', 'Category', 'Type', 'Amount'
        columns_to_keep = ['Date', 'Name', 'Category', 'Type', 'Amount']
        new_data = new_data[columns_to_keep]

        # Fill missing values in 'Amount' with 0
        new_data['Amount'] = new_data['Amount'].fillna(0)

        # Replace any non-numeric characters and convert to numeric
        new_data['Amount'] = pd.to_numeric(
            new_data['Amount'].str.replace(r'[^\d.]', '', regex=True),
            errors='coerce').fillna(0)

        # Overwrite existing CSV to avoid duplicates
        new_data.to_csv(DATA_FILE, index=False)
        logging.debug("File uploaded successfully.")
        return jsonify({"message": "File uploaded successfully"})
    except Exception as e:
        logging.error(f"Error in upload_file: {e}")
        return jsonify({"error": str(e)})


# Run Clustering
@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    if not os.path.exists(DATA_FILE):
        return jsonify(
            {"error": "No data available. Please upload a file first."})
    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)

        # Feature Engineering
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # Select columns to use for clustering
        columns_for_clustering = ['Amount', 'DayOfWeek', 'IsWeekend'] # Example columns that are numeric

        if 'Category' in data.columns:
            data['Category'].fillna('Unknown',
                                    inplace=True)  # Fill missing values with a placeholder
            le_category = LabelEncoder()
            data['CategoryEncoded'] = le_category.fit_transform(data['Category'])
            columns_for_clustering.append('CategoryEncoded')

        # Encoding 'Type'
        if 'Type' in data.columns:
            data['Type'].fillna('Unknown',
                                inplace=True)  # Fill missing values with a placeholder
            le_type = LabelEncoder()
            data['TypeEncoded'] = le_type.fit_transform(data['Type'])
            columns_for_clustering.append('TypeEncoded')

        # Use only numeric columns for clustering
        clustering_data = data[columns_for_clustering]

        # Handle missing values by filling them with the median
        clustering_data.fillna(clustering_data.median(), inplace=True)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)

        # KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_features)

        # Save updated data
        data.to_csv(DATA_FILE, index=False)
        logging.debug("Clustering successfully performed.")
        return jsonify({"message": "Clustering successfully performed"})
    except Exception as e:
        logging.error(f"Error in run_clustering: {e}")
        return jsonify({"error": str(e)})

# Get Insights
@app.route('/insights', methods=['GET'])
def get_insights():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."})
    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)

        # Convert 'Amount' column to numeric with error handling
        data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)

        # Check if required columns exist
        if 'Cluster' not in data.columns or 'DayOfWeek' not in data.columns:
            logging.error("Required columns are missing. Please run clustering first.")
            return jsonify({"error": "Required columns are missing. Please run clustering first."})

        # Log data preview for debugging
        logging.debug(f"Data preview: {data.head()}")

        # Group by cluster and calculate summary
        cluster_summary = data.groupby('Cluster').agg(
            Average_Amount=('Amount', 'mean'),
            Median_Amount=('Amount', 'median'),
            Count=('Amount', 'size'),
            Weekend_Spend_Percentage=('DayOfWeek', lambda x: ((x >= 5).sum() / len(x)) * 100 if len(x) > 0 else 0)
        ).reset_index()

        # Replace problematic values in cluster_summary for JSON serialization
        cluster_summary.replace([np.inf, -np.inf], 0, inplace=True)
        cluster_summary.fillna(0, inplace=True)  # Replacing NaN with 0 for better JSON serialization

        # Convert cluster_summary to dictionary
        cluster_summary = cluster_summary.to_dict(orient='records')

        # Clean detailed data for JSON serialization
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna('', inplace=True)  # Fill NaN values with an empty string to prevent JSON issues
        detailed_data = data.to_dict(orient='records')

        logging.debug(f"Cluster Summary: {cluster_summary}")

        return jsonify({"cluster_summary": cluster_summary, "data": detailed_data})
    except Exception as e:
        logging.error(f"Error in get_insights: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
