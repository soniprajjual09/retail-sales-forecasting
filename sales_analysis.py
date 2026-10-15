
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

print("Loading the full training dataset...")
df = pd.read_csv("data/train.csv")

df["date"] = pd.to_datetime(df["date"])

print("\nDataset shape:", df.shape)
print("Start date:", df["date"].min().date())
print("End date:", df["date"].max().date())
print("Number of stores:", df["store_nbr"].nunique())
print("Number of product families:", df["family"].nunique())

# Calculate total sales for each date
daily_sales = df.groupby("date")["sales"].sum()

print("\nFirst five daily sales totals:")
print(daily_sales.head())

# Plot daily sales
plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Retail Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/daily_sales.png")
plt.show()

# Plot sales for the top 10 product families
family_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 5))
family_sales.sort_values().plot(kind="barh")
plt.title("Top 10 Product Families by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Family")
plt.tight_layout()
plt.savefig("outputs/top_product_families.png")
plt.show()

print("\nAnalysis completed! Check the outputs folder.")
