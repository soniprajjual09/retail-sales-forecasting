
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

Path("outputs").mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])

daily = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["lag_1"] = daily["sales"].shift(1)
daily["lag_7"] = daily["sales"].shift(7)
daily["rolling_7"] = daily["sales"].shift(1).rolling(7).mean()

daily = daily.dropna().reset_index(drop=True)

features = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "rolling_7"
]

split = int(len(daily) * 0.8)

train = daily.iloc[:split]
test = daily.iloc[split:]

X_train = train[features]
X_test = test[features]
y_train = train["sales"]
y_test = test["sales"]

print("Evaluating models...")

predictions = {
    "Seasonal Naive": test["lag_7"].to_numpy()
}

linear = LinearRegression()
linear.fit(X_train, y_train)
predictions["Linear Regression"] = linear.predict(X_test)

forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
forest.fit(X_train, y_train)
predictions["Random Forest"] = forest.predict(X_test)

rows = []

for name, pred in predictions.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    rows.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

results = pd.DataFrame(rows).sort_values("MAE")
print("\n--- MODEL COMPARISON ---")
print(results.to_string(index=False))

results.to_csv("outputs/model_comparison.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["MAE"])
plt.title("Model Comparison: Mean Absolute Error")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()

print("\nSaved comparison table and graph in outputs.")
