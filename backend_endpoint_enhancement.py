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

## Helper function to preprocess data
def preprocess_data(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Feature: Day of the Week
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Feature: IsWeekend (1 if Saturday or Sunday, else 0)
    data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Handle missing values for 'Category' and 'Type'
    data['Category'] = data['Category'].fillna('Unknown')
    data['Type'] = data['Type'].fillna('Unknown')

    # Encode 'Category' and 'Type' as numerical features for clustering
    from sklearn.preprocessing import LabelEncoder

    le_category = LabelEncoder()
    le_type = LabelEncoder()

    data['CategoryEncoded'] = le_category.fit_transform(data['Category'])
    data['TypeEncoded'] = le_type.fit_transform(data['Type'])

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


# Enhanced Insights with Cluster Characteristics and Suggestions
@app.route('/insights', methods=['GET'])
def get_insights():
    if not os.path.exists(DATA_FILE):
        return jsonify(
            {"error": "No data available. Please upload a file first."})

    try:
        # Load the data
        data = pd.read_csv(DATA_FILE)

        # Initialize an empty list to store insights for each cluster
        cluster_insights = []

        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]

            # Calculate cluster characteristics
            average_amount = cluster_data['Amount'].mean()
            median_amount = cluster_data['Amount'].median()
            transaction_count = len(cluster_data)
            weekend_spend_percentage = cluster_data['IsWeekend'].mean() * 100
            top_categories = cluster_data['Category'].value_counts().head(
                3).to_dict()

            # Generate a suggestion based on spending behavior
            if average_amount > data['Amount'].quantile(0.75):
                suggestion = "This cluster has high-value transactions. Consider budgeting for these significant expenses."
            elif weekend_spend_percentage > 50:
                suggestion = "High percentage of weekend spending detected. Monitor leisure and social spending."
            elif transaction_count > data['Amount'].count() * 0.2:
                suggestion = "This cluster has frequent, low-value transactions. Consider tracking daily expenses closely."
            else:
                suggestion = "This cluster has a balanced spending pattern."

            # Append cluster insights
            cluster_insights.append({
                "Cluster": int(cluster),
                "Average Amount": average_amount,
                "Median Amount": median_amount,
                "Transaction Count": transaction_count,
                "Weekend Spend Percentage": weekend_spend_percentage,
                "Top Categories": top_categories,
                "Suggestion": suggestion
            })

        # Convert insights into JSON
        insights = {
            "cluster_insights": cluster_insights,
            "top_transactions": data.groupby('Cluster').apply(
                lambda x: x.nlargest(5, 'Amount')[
                    ['Date', 'Name', 'Category', 'Amount', 'Cluster']]
            ).reset_index(drop=True).to_dict(orient='records'),
            "top_frequent_expenses": data.groupby(
                ['Cluster', 'Category']).size()
            .reset_index(name='Frequency')
            .sort_values(['Cluster', 'Frequency'], ascending=[True, False])
            .groupby('Cluster').head(3).reset_index(drop=True).to_dict(
                orient='records'),
            "outliers": [
                {
                    "Cluster": int(cluster),
                    "High Outliers": cluster_data[cluster_data['Amount'] > (
                                cluster_data['Amount'].quantile(0.75) + 1.5 * (
                                    cluster_data['Amount'].quantile(0.75) -
                                    cluster_data['Amount'].quantile(0.25)))][
                        ['Date', 'Name', 'Category', 'Amount']].to_dict(
                        orient='records'),
                    "Low Outliers": cluster_data[cluster_data['Amount'] < (
                                cluster_data['Amount'].quantile(0.25) - 1.5 * (
                                    cluster_data['Amount'].quantile(0.75) -
                                    cluster_data['Amount'].quantile(0.25)))][
                        ['Date', 'Name', 'Category', 'Amount']].to_dict(
                        orient='records')
                } for cluster in data['Cluster'].unique()
            ]
        }

        return jsonify(insights)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)