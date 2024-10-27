from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

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
        # Overwrite existing CSV to avoid duplicates
        new_data.to_csv(DATA_FILE, index=False)
        return jsonify({"message": "File uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)})


# Run Clustering
@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    if not os.path.exists(DATA_FILE):
        return jsonify(
            {"error": "No data available. Please upload a file first."})
    try:
        # Load the data and perform clustering (dummy example for now)
        data = pd.read_csv(DATA_FILE)

        # Add 'DayOfWeek' column
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.dayofweek

        # Placeholder for clustering logic
        data['Cluster'] = (data.index % 3)  # Dummy clustering
        data.to_csv(DATA_FILE, index=False)
        return jsonify({"message": "Clustering successfully performed"})
    except Exception as e:
        return jsonify({"error": str(e)})


# Get Insights
@app.route('/insights', methods=['GET'])
def get_insights():
    if not os.path.exists(DATA_FILE):
        return jsonify(
            {"error": "No data available. Please upload a file first."})
    try:
        data = pd.read_csv(DATA_FILE)
        cluster_summary = data.groupby('Cluster').agg(
            Average_Amount=('Amount', 'mean'),
            Median_Amount=('Amount', 'median'),
            Count=('Amount', 'size'),
            Weekend_Spend_Percentage=(
            'DayOfWeek', lambda x: ((x >= 5).sum() / len(x)) * 100)
        ).reset_index().to_dict(orient='records')

        detailed_data = data.to_dict(orient='records')
        return jsonify(
            {"cluster_summary": cluster_summary, "data": detailed_data})
    except Exception as e:
        return jsonify({"error": str(e)})


# Clear Data
@app.route('/clear_data', methods=['POST'])
def clear_data():
    try:
        # Clear the CSV by writing an empty DataFrame
        pd.DataFrame().to_csv(DATA_FILE, index=False)
        return jsonify({"message": "Data cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
