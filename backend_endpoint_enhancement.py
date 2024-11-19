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

import numpy as np

def convert_to_json_serializable(obj):
    """Recursively convert objects to JSON serializable types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)  # Convert timestamps or periods to strings
    elif isinstance(obj, dict):
        return {str(key): convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    return obj

def process_data(data):
    try:
        # Validate and convert 'Date' column
        if 'Date' in data.columns:
            # Convert 'Date' to datetime, setting errors='coerce' to handle invalid values
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            if data['Date'].isna().any():
                logging.warning("Some 'Date' values could not be converted. Dropping these rows.")
            data.dropna(subset=['Date'], inplace=True)  # Drop rows where 'Date' is NaT
            data['DayOfWeek'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
            data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
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
    cluster_distributions = {}
    for cluster in data['Cluster'].unique():
        cluster_distributions[str(cluster)] = data[data['Cluster'] == cluster]['Amount'].tolist()
    return cluster_distributions

def summarize_cluster(data):
    cluster_summaries = []
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        # Calculate spending per category
        category_spending = cluster_data.groupby('Category')['Amount'].sum().to_dict()

        # Identify biggest category
        biggest_category = max(category_spending, key=category_spending.get)

        # Highlight most recurring transaction in the biggest category
        recurring_transaction = (
            cluster_data[cluster_data['Category'] == biggest_category]
            .groupby(['Name'])
            ['Amount'].sum().sort_values(ascending=False).head(1).to_dict()
        )

        cluster_summaries.append({
            "Cluster": cluster,
            "Category Spending": category_spending,
            "Cluster Type": "Routine daily spending" if len(cluster_data) > 50 else "Occasional big-ticket spending",
            "Biggest Category": biggest_category,
            "Recurring Transaction": recurring_transaction
        })
    return cluster_summaries

def spending_by_time_period(data):
    """Categorize spending into time periods of the month."""
    time_periods = {"Early": [], "Middle": [], "End": []}

    # Ensure 'Day' column exists
    data['Day'] = pd.to_datetime(data['Date']).dt.day

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        early_spending = cluster_data[cluster_data['Day'] <= 10].groupby('Category')['Amount'].sum()
        middle_spending = cluster_data[(cluster_data['Day'] > 10) & (cluster_data['Day'] <= 20)].groupby('Category')['Amount'].sum()
        end_spending = cluster_data[cluster_data['Day'] > 20].groupby('Category')['Amount'].sum()

        time_periods['Early'].append({str(key): value for key, value in early_spending.to_dict().items()})
        time_periods['Middle'].append({str(key): value for key, value in middle_spending.to_dict().items()})
        time_periods['End'].append({str(key): value for key, value in end_spending.to_dict().items()})

    return time_periods

def calculate_monthly_spending_with_details(data):
    """
    Calculate monthly spending for each cluster with detailed transactions for the biggest category.
    :param data: DataFrame containing spending data with 'Cluster', 'Date', 'Category', and 'Amount' columns.
    :return: Dictionary with clusters as keys, including monthly spending and details for the biggest category.
    """
    # Ensure the 'Date' column is a datetime object
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Add 'Month' for grouping
    data['Month'] = data['Date'].dt.month

    monthly_spending = {}

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        # Group by Month and Category, sum the spending
        spending_by_month_category = cluster_data.groupby(['Month', 'Category'])['Amount'].sum().unstack(fill_value=0)

        # Identify the biggest category for the cluster based on total spending
        total_category_spending = cluster_data.groupby('Category')['Amount'].sum()
        biggest_category = total_category_spending.idxmax() if not total_category_spending.empty else None

        # Collect detailed transactions for the biggest category
        category_transactions = cluster_data[cluster_data['Category'] == biggest_category][['Date', 'Name', 'Amount']].to_dict(orient='records')

        # Structure monthly spending data
        monthly_spending[cluster] = {
            "monthly_data": [
                spending_by_month_category.loc[month].to_dict() if month in spending_by_month_category.index else {}
                for month in range(1, 13)
            ],
            "biggest_category": biggest_category,
            "transactions": category_transactions
        }
    return monthly_spending

def calculate_monthly_spending(data):
    """Calculate monthly spending for each cluster."""
    monthly_spending = {}

    # Ensure 'Month' column exists
    data['Month'] = pd.to_datetime(data['Date']).dt.to_period('M')

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        monthly_totals = cluster_data.groupby(['Month', 'Category'])['Amount'].sum()

        # Convert the MultiIndex DataFrame to a JSON-compatible dictionary
        monthly_spending[str(cluster)] = {
            str(month): {str(category): float(amount) for category, amount in categories.items()}
            for month, categories in monthly_totals.unstack(fill_value=0).iterrows()
        }

    return monthly_spending

def calculate_category_spending(data):
    """
    Calculate category spending for each cluster.
    :param data: DataFrame containing spending data with 'Cluster', 'Category', and 'Amount' columns.
    :return: Dictionary with clusters as keys and category spending data.
    """
    category_spending = {}

    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]

        # Calculate total spending by category
        category_totals = cluster_data.groupby('Category')['Amount'].sum()

        # If no category data, initialize an empty dictionary for the cluster
        category_spending[str(cluster)] = category_totals.to_dict() if not category_totals.empty else {}

    return category_spending

def calculate_high_value_transactions(data):
    high_value_summary = {}
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        highest_transactions = cluster_data.nlargest(3, 'Amount')[['Date', 'Name', 'Category', 'Amount']]
        high_value_summary[cluster] = highest_transactions.to_dict(orient='records')
    return high_value_summary


@app.route('/insights', methods=['GET'])
def get_insights():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data available. Please upload a file first."}), 400

    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)
        logging.info(f"Data loaded for insights with columns: {data.columns.tolist()}")

        if 'Cluster' not in data.columns:
            return jsonify({"error": "No clustering information available. Please run clustering first."}), 400

        # Generate insights
        cluster_summaries = summarize_cluster(data)
        transaction_distributions = calculate_transaction_distribution(data)
        time_period_spending = spending_by_time_period(data)
        monthly_spending = calculate_monthly_spending(data)

        # Ensure all outputs are JSON serializable
        insights = {
            "cluster_summaries": cluster_summaries,
            "transaction_distributions": transaction_distributions,
            "time_period_spending": time_period_spending,
            "monthly_spending": monthly_spending
        }

        # Convert the entire response to JSON serializable format
        return jsonify(convert_to_json_serializable(insights)), 200

    except Exception as e:
        logging.error(f"Error in get_insights: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)