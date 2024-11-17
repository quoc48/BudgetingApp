import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

app = Flask(__name__)
CORS(app)

DATA_FILE = 'spending_data.csv'

# Upload Data
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        new_data = pd.read_csv(file)
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

def convert_numpy(obj):
    """Recursively convert numpy types in JSON data to Python-native types."""
    if isinstance(obj, dict):
        return {str(key): convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return obj.tolist()
    else:
        return obj

# Helper function to preprocess data
def process_data(data):
    try:
        # Validate and convert 'Date' column
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            if data['Date'].isna().any():
                logging.warning("Some 'Date' values could not be converted. They will be dropped.")
            data.dropna(subset=['Date'], inplace=True)
            data['DayOfWeek'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
            data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            logging.info(f"Derived 'DayOfWeek' and 'IsWeekend' features successfully.")
        else:
            raise ValueError("The 'Date' column is missing and required for processing.")

        # Validate and clean 'Amount' column
        if 'Amount' in data.columns:
            data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)
        else:
            raise ValueError("The 'Amount' column is missing and required for clustering.")

        # Fill missing values for 'Category' and 'Type'
        data['Category'] = data['Category'].fillna('Unknown')
        data['Type'] = data['Type'].fillna('Unknown')

        logging.info(f"Processed data preview:\n{data.head()}")
        return data
    except Exception as e:
        logging.error(f"Error in process_data: {e}")
        raise

# Run clustering
@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."}), 400

    try:
        data = pd.read_csv(DATA_FILE)
        logging.info(f"Data loaded for clustering with columns: {data.columns.tolist()}")

        # Process data
        data = process_data(data)

        # Check required features
        required_features = ['Amount', 'DayOfWeek', 'IsWeekend']
        missing_features = [feature for feature in required_features if feature not in data.columns]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # Perform clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[required_features])
        kmeans = KMeans(n_clusters=6, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_features)

        # Log cluster counts
        logging.info(f"Cluster counts: {data['Cluster'].value_counts()}")

        # Save updated data
        data.to_csv(DATA_FILE, index=False)
        logging.info("Clustered data successfully saved.")

        return jsonify({"message": "Clustering successfully performed"}), 200

    except Exception as e:
        logging.error(f"Error in run_clustering: {e}")
        return jsonify({"error": str(e)}), 500

def calculate_transaction_distribution(data):
    """Calculate transaction distributions for each cluster."""
    cluster_distributions = {}
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        cluster_distributions[int(cluster)] = cluster_data['Amount'].tolist()  # Ensure cluster is an int for JSON compatibility
    return cluster_distributions

def summarize_cluster(data):
    """Summarize each cluster's characteristics with deeper behavioral analysis."""
    cluster_summaries = []
    total_transactions = data['Amount'].count()

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        # Core summary statistics
        avg_amount = cluster_data['Amount'].mean()
        median_amount = cluster_data['Amount'].median()
        transaction_count = len(cluster_data)
        weekend_percentage = cluster_data['IsWeekend'].mean() * 100
        most_common_categories = cluster_data['Category'].value_counts().head(
            3).to_dict()
        monthly_spending = (
            cluster_data.groupby(cluster_data['Date'].str[:7])['Amount'].sum().to_dict()
        )

        # Behavior insights
        if weekend_percentage > 50:
            cluster_type = "Weekend-focused spending"
        elif transaction_count > total_transactions * 0.2:
            if avg_amount < data['Amount'].mean():
                cluster_type = "Low-value frequent transactions"
            else:
                cluster_type = "High-value frequent transactions"
        else:
            cluster_type = "Occasinal big_ticket spending"

        cluster_summaries.append({
            "Cluster": int(cluster),
            "Average Amount": avg_amount,
            "Median Amount": median_amount,
            "Transaction Count": transaction_count,
            "Weekend Percentage": weekend_percentage,
            "Most Common Categories": most_common_categories,
            "Monthly Spending": monthly_spending,
            "Cluster Type": cluster_type
        })
    return cluster_summaries

def spending_by_time_period(data):
    """Categorize spending into time periods of the month."""
    time_periods = {"Early": [], "Middle": [], "End": []}

    # Ensure 'Day' column exists in the original DataFrame
    if 'Day' not in data.columns:
        data['Day'] = pd.to_datetime(data['Date']).dt.day

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        early_spending = cluster_data[cluster_data['Day'] <= 10].groupby('Category')['Amount'].sum()
        middle_spending = cluster_data[(cluster_data['Day'] > 10) & (cluster_data['Day'] <= 20)].groupby('Category')['Amount'].sum()
        end_spending = cluster_data[cluster_data['Day'] > 20].groupby('Category')['Amount'].sum()

        # Ensure JSON serialization by converting indexes to strings
        time_periods['Early'].append(early_spending.to_dict())
        time_periods['Middle'].append(middle_spending.to_dict())
        time_periods['End'].append(end_spending.to_dict())

    return time_periods


@app.route('/insights', methods=['GET'])
def get_insights():
    # Ensure data is available
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."}), 400

    try:
        # Load and process data
        data = pd.read_csv(DATA_FILE)
        logging.info(
            f"Data loaded for insights with columns: {data.columns.tolist()}")

        if 'Cluster' not in data.columns:
            return jsonify({"error": "No clustering information available. Please run clustering first."}), 400

        # Summarize clusters
        cluster_summaries = summarize_cluster(data)
        logging.info(f"Generated cluster summaries: {cluster_summaries}")

        # Calculate transaction distributions
        transaction_distributions = calculate_transaction_distribution(data)
        logging.info(f"Transaction distributions: {transaction_distributions}")

        time_period_spending = spending_by_time_period(data)
        logging.info(f"Spending by time period: {time_period_spending}")

        # Include it in the response
        insights = {
            "cluster_summaries": cluster_summaries,
            "transaction_distributions": transaction_distributions,
            "time_period_spending": time_period_spending
        }

        return jsonify(insights), 200

    except Exception as e:
        logging.error(f"Error in get_insights: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)