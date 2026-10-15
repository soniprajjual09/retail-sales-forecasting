
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

Path("outputs").mkdir(exist_ok=True)

print("Loading historical sales...")
df = pd.read_csv("data/train.csv")
df["date"] = pd.to_datetime(df["date"])

daily = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
    .reset_index(drop=True)
)

daily["day_of_week"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["day_of_year"] = daily["date"].dt.dayofyear
daily["lag_1"] = daily["sales"].shift(1)
daily["lag_7"] = daily["sales"].shift(7)
daily["rolling_7"] = daily["sales"].shift(1).rolling(7).mean()

features = [
    "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "rolling_7"
]

training = daily.dropna().copy()

X_train = training[features]
y_train = training["sales"]

print("Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Store historical daily sales for recursive forecasting
history = daily["sales"].tolist()
last_date = daily["date"].max()

future_rows = []

for i in range(1, 8):
    forecast_date = last_date + pd.Timedelta(days=i)

    row = pd.DataFrame([{
        "day_of_week": forecast_date.dayofweek,
        "month": forecast_date.month,
        "day_of_year": forecast_date.dayofyear,
        "lag_1": history[-1],
        "lag_7": history[-7],
        "rolling_7": np.mean(history[-7:])
    }])

    predicted_sales = float(model.predict(row[features])[0])

    future_rows.append({
        "date": forecast_date,
        "predicted_sales": predicted_sales
    })

    # Use the prediction as input for the next forecast day
    history.append(predicted_sales)

forecast = pd.DataFrame(future_rows)
forecast["predicted_sales"] = forecast["predicted_sales"].clip(lower=0)

forecast.to_csv("outputs/next_7_days_forecast.csv", index=False)

print("\n--- NEXT 7 DAYS SALES FORECAST ---")
print(forecast.to_string(index=False, formatters={
    "predicted_sales": "{:,.2f}".format
}))

plt.figure(figsize=(10, 5))
plt.plot(
    daily["date"].tail(30),
    daily["sales"].tail(30),
    label="Historical Sales"
)
plt.plot(
    forecast["date"],
    forecast["predicted_sales"],
    marker="o",
    label="Forecast Sales"
)
plt.title("Retail Sales Forecast for the Next 7 Days")
plt.xlabel("Date")
plt.ylabel("Total Daily Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/next_7_days_forecast.png")
plt.show()

print("\nForecast saved in the outputs folder.")
