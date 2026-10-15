
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

Path("outputs").mkdir(exist_ok=True)

print("Loading sales data...")

df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])

# Convert store and product-level data into total daily sales
daily = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

# Create features from past sales and calendar information
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["sales_lag_1"] = daily["sales"].shift(1)
daily["sales_lag_7"] = daily["sales"].shift(7)
daily["sales_rolling_7"] = daily["sales"].shift(1).rolling(7).mean()

daily = daily.dropna().reset_index(drop=True)

features = [
    "day_of_week",
    "month",
    "day_of_year",
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_7"
]

X = daily[features]
y = daily["sales"]

# Split chronologically: earlier dates train, later dates test
split = int(len(daily) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]
test_dates = daily["date"].iloc[split:]

print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n--- MODEL RESULTS ---")
print("Training days:", len(X_train))
print("Testing days:", len(X_test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save actual and predicted values
results = pd.DataFrame({
    "date": test_dates.to_numpy(),
    "actual_sales": y_test.to_numpy(),
    "predicted_sales": predictions
})

results.to_csv("outputs/forecast_results.csv", index=False)

# Plot actual vs predicted sales for the test period
plt.figure(figsize=(14, 6))
plt.plot(results["date"], results["actual_sales"], label="Actual Sales")
plt.plot(results["date"], results["predicted_sales"], label="Predicted Sales")
plt.title("Retail Sales Forecast: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

print("\nForecasting complete!")
print("Results saved in the outputs folder.")
