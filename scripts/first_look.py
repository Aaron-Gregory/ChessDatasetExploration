"""
This is a script to get some initial feeling for the chess_games dataset.

Loads in the csv file and prints a variety of basic information:
* Number of rows and columns
* Summaries for both categorical and numerical columns
* Number of missing values
"""

import pandas as pd

# Load the dataset
df = pd.read_csv("./data/chess_games.csv")

# Dataset shape
print("Dataset shape:")
print(df.shape)

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Unique values in categorical columns
print("\nUnique values in categorical columns:")
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    unique_vals = df[col].unique()
    print(f"    {col:20s}: {len(unique_vals):5d}   {unique_vals[:3]}...")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
