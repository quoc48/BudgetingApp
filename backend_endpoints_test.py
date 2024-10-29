import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import logging

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
        # Load the data and perform clustering (dummy example for now)
        data = pd.read_csv(DATA_FILE)

        # Add 'DayOfWeek' column
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['DayOfWeek'] = data['Date'].dt.dayofweek

        # Convert 'Amount' column to numeric
        data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(
            0)

        # Placeholder for clustering logic
        data['Cluster'] = (data.index % 3)  # Dummy clustering
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

# Send Monthly Data
@app.route('/monthly_data', methods=['GET'])
def monthly_data():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."})
    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)

        # Convert 'Date' column to datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Drop rows where 'Date' could not be converted
        data = data.dropna(subset=['Date'])

        # Extract month and year for grouping
        data['Month_Year'] = data['Date'].dt.to_period('M')

        # Group by 'Month_Year' and sum the 'Amount' column
        monthly_spend = data.groupby('Month_Year').agg({'Amount': 'sum'}).reset_index()

        # Convert 'Month_Year' back to string for easier handling in frontend
        monthly_spend['Month_Year'] = monthly_spend['Month_Year'].astype(str)

        # Prepare response data
        response_data = {
            "months": monthly_spend['Month_Year'].tolist(),
            "spendings": monthly_spend['Amount'].tolist()
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
