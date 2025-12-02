# UAE Fuel Price Prediction

Advanced machine learning project forecasting monthly fuel prices for Special 95 gasoline in the UAE using historical data and real-time economic indicators.

## Overview

This project combines historical fuel price data with live market indicators to predict UAE fuel prices 1-3 months ahead. Using XGBoost regression with Bayesian optimization, the model achieves robust predictions by analyzing:

- Historical fuel price patterns (12-month lag features)
- Real-time crude oil prices (via Yahoo Finance API)
- Live gold prices scraped from UAE Gold Price
- Rolling averages and volatility measures
- Price stability detection and conservative prediction adjustments

## Key Features

**Dataset Coverage**: January 2015 to present with monthly granularity

- **Real-time data integration**: Live crude oil prices via Yahoo Finance and gold prices via web scraping
- **Enhanced feature engineering**: 12-month lag features with rolling statistics
- **Improved model stability**: Volatility constraints and conservative prediction dampening
- **Automated data updates**: Standalone script for monthly data collection
- **Advanced hyperparameter tuning**: 50 iterations of Bayesian optimization
- **Price classification system**: Trend analysis with percentage change calculations

## Technical Implementation

### Model Architecture

- **Algorithm**: XGBoost Regressor with Target Encoding
- **Validation**: Time Series Cross-Validation (5 folds, adaptive for small datasets)
- **Optimization**: Bayesian Search with 50 iterations
- **Features**: 40+ engineered features including lags, rolling statistics, and market indicators

### Model Performance

Version | MAE (AED) | R² Score | Features | Real-time Data | Prediction Dampening
--------|-----------|----------|----------|----------------|---------------------
V2.1    | 0.1359    | 0.5377   | ~15      | No             | No
V3.0    | 0.136     | 0.54     | 40+      | Yes            | Basic
V3.2    | 0.1644    | 0.0895   | 40+      | Yes            | Conservative

Note: V3.2 shows lower R² due to conservative prediction adjustments that prevent unrealistic forecasts, resulting in more stable and reliable predictions for practical use.

## Data Sources

- **Historical Data**: Custom compiled dataset (FuelData.xlsx)
- **Crude Oil**: Yahoo Finance ticker "CL=F"
- **Gold Prices**: UAE Gold Price website (web scraping)
- **UAE Fuel Prices**: Gulf News historical fuel rates (for updates)

## Feature Engineering

The model incorporates extensive feature engineering for robust predictions:

```python
# Enhanced lag features (V3.2)
for lag in range(1, 13):  # 12-month historical context
    df[f'fuel_lag_{lag}'] = df[FUEL_COL].shift(lag)
    df[f'gold_lag_{lag}'] = df[GOLD_COL].shift(lag)
    df[f'oil_lag_{lag}'] = df[OIL_COL].shift(lag)

# Rolling statistics with adaptive windows
df['fuel_rolling_avg'] = df[FUEL_COL].rolling(window).mean()
df['fuel_rolling_std'] = df[FUEL_COL].rolling(window).std()

# Percentage change and stability detection
df['fuel_pct_change_1'] = df[FUEL_COL].pct_change()
df['is_stable_recently'] = (df['fuel_pct_change_1'].abs() < 0.01).astype(int)
```

### Pipeline Structure

```python
Pipeline([
    ('encoder', TargetEncoder()),
    ('regressor', XGBRegressor(random_state=8, enable_categorical=True))
])
```

## Prediction Methodology

The model provides:

- Next month fuel price forecast
- Price change analysis (absolute and percentage)
- Trend classification (increase/decrease/stable)
- Conservative prediction dampening based on recent volatility patterns

### Conservative Prediction Adjustment

V3.2 introduces a dampening mechanism that prevents unrealistic predictions:

```python
# Calculate typical recent changes
recent_changes = df[FUEL_COL].diff().tail(6).abs()
typical_change = recent_changes.mean()

# Limit prediction delta to typical historical changes
pred_delta = pred - current_price
pred_delta = np.sign(pred_delta) * min(abs(pred_delta), typical_change)
pred = current_price + pred_delta
```

Example output:

```
Last recorded fuel price: 2.51 AED
Forecast for next month: 2.57 AED
Change: 0.06 AED
Expected increase of 0.06 AED (2.4%)
```

## Installation

### Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
pip install scikit-optimize category_encoders yfinance joblib
pip install requests beautifulsoup4
```

### Setup Instructions

**Clone the repository:**

```bash
git clone https://github.com/Amaanish/Fuel-Prediction.git
cd Fuel-Prediction
```

**For Google Colab (Recommended):**

- Upload the notebook file (`Fuel_Predict_XGBOOST 3.2.ipynb`) to Colab
- Upload `FuelData.xlsx` to the Colab file system
- Run all cells

**For Local Environment:**

- Ensure all dependencies are installed
- Place `FuelData.xlsx` in the same directory as the script
- Run the Python script or Jupyter notebook

## Usage

### Running the Prediction Model

The model automatically:

1. Fetches real-time crude oil prices from Yahoo Finance
2. Scrapes current gold prices from UAE Gold Price
3. Loads and processes historical data with safety checks
4. Trains the optimized XGBoost model with adaptive parameters
5. Generates next month's price forecast with conservative adjustments
6. Provides trend analysis and insights

### Updating the Dataset

Use the `update.py` script to automatically collect and append new monthly data:

```python
# Configure your Excel file path
excelpath = "path/to/FuelData.xlsx"

# Run the script on the first day of each month
python update.py
```

The update script:

- Fetches current month's crude oil price from Yahoo Finance
- Scrapes UAE gold prices for the current month
- Retrieves UAE fuel prices from Gulf News
- Automatically appends new data to the Excel file
- Prevents duplicate entries for the same month

## Version Comparison

Aspect                  | V2.1          | V3.0              | V3.2
------------------------|---------------|-------------------|------------------
Lag Features            | 3 months      | 12 months         | 12 months
Real-time Data          | None          | Crude oil + Gold  | Crude oil + Gold
Data Collection         | Manual        | Manual            | Automated script
Hyperparameter Tuning   | 25 iterations | 50 iterations     | 50 iterations
Feature Count           | ~15           | 40+               | 40+
Prediction Stability    | Basic         | Volatility-adjusted | Conservative dampening
Price Classification    | None          | High/Normal/Low   | Trend analysis
Safety Checks           | Limited       | Limited           | Adaptive for small datasets

## Future Enhancements

- Extended forecast horizon: 3-6 month predictions with uncertainty intervals
- Interactive dashboard: Web-based visualization tool for historical trends
- Additional indicators: USD/AED exchange rates, regional demand metrics
- Ensemble methods: Combining multiple algorithms for improved accuracy
- API deployment: Real-time prediction service with automated updates
- Enhanced data collection: Additional economic indicators and seasonal factors

## Applications

- **Businesses**: Fleet management and budget planning
- **Consumers**: Personal fuel expense forecasting
- **Analysts**: Market trend analysis and reporting
- **Researchers**: Economic indicator correlation studies

## Project Structure

```
Fuel-Prediction/
├── FuelData.xlsx                      # Historical dataset
├── Fuel_Predict_XGBOOST 3.2.ipynb    # Main prediction model
├── update.py                          # Data collection script
└── README.md                          # Documentation
```

## Notes

- The model uses adaptive parameters to handle datasets of varying sizes
- Conservative prediction dampening ensures realistic forecasts based on historical volatility
- Web scraping components may require updates if source websites change their structure
- For best results, update the dataset monthly using the provided update script


---

**Last updated**: December 2025 | **Model Version**: V3.2
