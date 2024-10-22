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

