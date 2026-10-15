
import pandas as pd

file_path = "data/train.csv"

df = pd.read_csv(file_path, nrows=10000)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET INFORMATION ---")
print("Number of rows loaded:", len(df))
print("Columns:", df.columns.tolist())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- SALES SUMMARY ---")
print(df["sales"].describe())

df["date"] = pd.to_datetime(df["date"])
print("\nDate range:", df["date"].min(), "to", df["date"].max())

print("\nData loaded successfully!")
