import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("data/spending_data.csv")

# Step 2: Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Feature extraction
# Extract day of the week from the date
data['Date'] = pd.to_datetime(data['Date'])
data['Day0fWeek'] = data['Date'].dt.dayofweek

# Step 4: Select relevant feature for clustering
features = data[['Amount', 'Day0fWeek']]

# Step 5: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Output the preprocessed data
print("Preprocessed Data:")
print(scaled_features)
