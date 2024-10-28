import pandas as pd
import re

# Load your CSV file
data = pd.read_csv('Expenses.csv')

# Define a function to remove links
def remove_links(text):
    if isinstance(text, str):
        return re.sub(r'\s*\(http\S+\)', '', text)
    return text

# Apply the function to the desired column (e.g., 'Category' or 'Name')
data['Category'] = data['Category'].apply(remove_links)
data['Type'] = data['Type'].apply(remove_links)

# Optionally, clean up any leftover parentheses or extra spaces
data['Category'] = data['Category'].str.replace(r'\s*\(\s*\)', '', regex=True).str.strip()
data['Type'] = data['Type'].str.replace(r'\s*\(\s*\)', '', regex=True).str.strip()

# Keep only the specified columns
columns_to_keep = ['Date', 'Name', 'Category', 'Type', 'Amount']
filtered_data = data[columns_to_keep]

# Format the 'Amount' column in Vietnamese currency
filtered_data['Amount'].fillna(0, inplace=True)
filtered_data['Amount'] = filtered_data['Amount'].apply(lambda x: f"{int(x):,}")

# Save the cleaned data back to a CSV
filtered_data.to_csv('Expense_cleaned.csv', index=False)
print("Links removed and data saved to 'Expense_cleaned.csv'")