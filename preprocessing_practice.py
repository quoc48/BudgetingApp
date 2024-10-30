import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = {
    'Age': [25, 45, 30, 54],
    'Salary': [40000, 80000, 60000, 120000]
}

df = pd.DataFrame(data)

# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=['Age', 'Salary'])
print("Standardized Data:")
print(standardized_df)

# Normalization
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=['Age', 'Salary'])
print("\nNormalized Data:")
print(normalized_df)

