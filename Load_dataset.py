import pandas as pd

# Replace 'your_dataset.csv' with your actual dataset file name
data_path = 'IMDB_Dataset.csv'

# Load dataset
df = pd.read_csv(data_path)

# Show first 5 rows of the dataset
print("Dataset preview:")
print(df.head())

# Display dataset info
print("\nColumn names:")
print(df.columns)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Print number of rows and columns
print("\nDataset shape:", df.shape)