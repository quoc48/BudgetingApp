from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# In-memory storage for spending data
data = pd.DataFrame()

# Endpoint 1: Upload spending data (CSV)
@app.route('/upload', methods=['POST'])
def upload_data():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        uploaded_data = pd.read_csv(file)
        data = pd.concat([data, uploaded_data], ignore_index=True)
        return jsonify({'message': 'File successfully uploaded'}), 200
    else:
        return jsonify({'error': 'Invalid file format, please upload a CSV'}), 400

# Endpoint 2: Run the clustering model
@app.route('/run_clustering', method=['POST'])
def run_clustering():
    global data
    if data.empty:
        return jsonify({'error': 'No data available for clustering'}), 400

    # Preprocessing data
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x>=5 else 0)
        le = LabelEncoder()
        data['CategoryEncoded'] = le.fit_transform(data['Category'])
    except KeyError as e:
        return jsonify({'error': f'Missing required columns in data: {str(e)}'}), 400

    # Select features for clustering
    features = data[['Amount', 'DayOfWeek', 'IsWeekend', 'CategoryEncoded']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Run K-means clustering
    best_k = 3
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(scaled_features)
    data['Cluster'] = kmeans.labels_

    return jsonify({'message': 'Clustering successfully performed', 'number_of_clusters': best_k}), 200

# Endpoint 3: Retrieve categorized data and insights
@app.route('/insights', methods=['GET'])
def get_insights():
    global data
    if data.empty:
        return jsonify({'error': 'No data available'}), 400

    # Generate insights
    cluster_summary = data.groupby('Cluster').agg(
        Average_Amount=('Amount', 'mean'),
        Median_Amount=('Amount', 'median'),
        Count=('Amount', 'size'),
        Weekend_Spend_Percentage=('IsWeekend', 'mean')).reset_index()
    cluster_summary['Weekend_Spend_Percentage'] *= 100

    response = {
        'cluster_summary':
            cluster_summary.to_dict(orient='records'),
        'data': data.to_dict(orient='records')
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)