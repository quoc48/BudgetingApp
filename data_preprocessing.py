import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("data/spending_data.csv")

# Step 2: Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Output the loaded data
print("Load Data:")
print(data)
