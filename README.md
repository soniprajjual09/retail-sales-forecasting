# Retail Sales Forecasting Using Machine Learning

## 1. Project Overview

Retail Sales Forecasting is a Predictive Analytics mini-project that uses historical retail sales data and machine learning algorithms to predict future sales.

The project analyzes sales trends over time, compares forecasting models, and generates a seven-day sales forecast based on historical data.

## 2. Objectives

- Analyze historical retail sales data.
- Identify daily sales trends and patterns.
- Build and compare machine learning forecasting models.
- Evaluate model performance using MAE and RMSE.
- Generate forecasts for the next seven days.

## 3. Dataset

**Dataset:** Store Sales – Time Series Forecasting  
**Source:** [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

The dataset contains historical sales records from multiple stores and product families.

The project uses daily aggregated sales and engineered features, including:
- Day of the week
- Month
- Day of the year
- Previous day's sales (lag 1)
- Sales from seven days earlier (lag 7)
- Previous seven-day rolling average

The dataset is not included in this repository because of its size. Place the required CSV files in the `data/` folder before running the scripts.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Git and GitHub
- Visual Studio Code

## 5. Machine Learning Models

The project compares three forecasting approaches:

1. **Linear Regression:** Models the relationship between engineered features and sales.
2. **Random Forest Regressor:** Combines multiple decision trees to predict sales.
3. **Seasonal Naive:** Uses sales from the corresponding day of the previous week as a baseline.

## 6. Model Evaluation Results

The models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

| Model | MAE | RMSE |
|---|---:|---:|
| Random Forest | 84,785.13 | 131,142.33 |
| Linear Regression | 87,760.30 | 139,218.32 |
| Seasonal Naive | 108,674.91 | 170,404.30 |

Random Forest achieved the lowest MAE and RMSE among the three tested approaches.

## 7. Project Structure

```text
Retail_Sales_Forecasting/
├── check_data.py
├── sales_analysis.py
├── forecast_model.py
├── compare_models.py
├── future_forecast.py
├── data/
├── outputs/
│   ├── daily_sales.png
│   ├── top_product_families.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   └── next_7_days_forecast.png
├── .gitignore
└── README.md
```

## 8. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/soniprajjual09/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### Step 2: Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Step 4: Add the dataset

Download the required CSV files from Kaggle and place them in the `data/` directory.

### Step 5: Run the scripts

```bash
python check_data.py
python sales_analysis.py
python forecast_model.py
python compare_models.py
python future_forecast.py
```

## 9. Limitations

- Forecast accuracy depends on the quality and coverage of historical data.
- The current model uses aggregated daily sales rather than forecasting each store and product family independently.
- The seven-day forecast is based on the dataset's historical endpoint, not current real-world sales.
- Additional features and more advanced time-series models may improve performance.

## 10. Future Scope

- Build separate forecasts for individual stores and product families.
- Include holidays, promotions, and transactions as additional predictive features.
- Experiment with advanced time-series models.
- Develop an interactive dashboard for visualizing sales forecasts.

## 11. Author

**Student:** Prajjual Soni  
**Registration Number:** 2427030080  
**Course:** CSE 3141 – Predictive Analytics  
**Project Guide:** Mr. Lav Upadhyay
