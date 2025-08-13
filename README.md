# UAE Fuel Price Forecasting - Special 95 Gasoline

Advanced machine learning project forecasting monthly fuel prices for Special 95 gasoline in the UAE using historical data and real-time economic indicators.

##  Project Overview

This project combines historical fuel price data with live market indicators to predict UAE fuel prices 1-3 months ahead. Using XGBoost regression with Bayesian optimization, the model achieves robust predictions by analyzing:

- **Historical fuel price patterns** (12-month lag features)
- **Real-time crude oil prices** (via Yahoo Finance API)
- **Live gold prices** (economic stability indicator)
- **Rolling averages and volatility measures**
- **Price stability detection**

**Dataset Coverage**: January 2015 to present with monthly granularity

##  Key Features

### V3 Enhancements (Latest)
- **Real-time data integration**: Live crude oil and gold prices via Yahoo Finance
- **Enhanced feature engineering**: 12-month lag features vs 3-month in V2.1
- **Improved model stability**: Volatility constraints and prediction smoothing
- **Better performance**: R² Score ~0.54 with robust MAE of 0.136 AED
- **Advanced hyperparameter tuning**: 50 iterations vs 25 in V2.1
- **Price classification system**: Categorizes current prices as high/normal/low

### Model Architecture
- **Algorithm**: XGBoost Regressor with Target Encoding
- **Validation**: Time Series Cross-Validation (5 folds)
- **Optimization**: Bayesian Search with 50 iterations
- **Features**: 40+ engineered features including lags, rolling statistics, and market indicators

##  Performance Metrics

| Version | MAE (AED) | R² Score | Features | Real-time Data |
|---------|-----------|----------|----------|----------------|
| V2.1    | 0.1359    | 0.5377   | ~15      | ❌             |
| V3      | 0.136     | 0.54     | 40+      | ✅             |

## 🛠️ Technical Implementation

### Data Sources
- **Historical Data**: Custom compiled dataset (FuelData.xlsx)
- **Crude Oil**: Yahoo Finance ticker "CL=F"
- **Gold Prices**: Yahoo Finance ticker "GC=F"

### Feature Engineering
```python
# Enhanced lag features (V3)
for lag in range(1, 13):  # 12 months vs 3 in V2.1
    df[f'fuel_lag_{lag}'] = df[FUEL_COL].shift(lag)
    df[f'gold_lag_{lag}'] = df[GOLD_COL].shift(lag)
    df[f'oil_lag_{lag}'] = df[OIL_COL].shift(lag)

# Rolling statistics with 12-month windows
df['fuel_rolling_avg_3'] = df[FUEL_COL].rolling(12).mean()
df['fuel_rolling_std_3'] = df[FUEL_COL].rolling(12).std()
```

### Model Pipeline
```python
Pipeline([
    ('encoder', TargetEncoder()),
    ('regressor', XGBRegressor(random_state=8, enable_categorical=True))
])
```

##  Prediction Capabilities

The model provides:
- **Next month fuel price forecast**
- **Price change analysis** (absolute and percentage)
- **Trend classification** (increase/decrease/stable)
- **Volatility adjustment** (prevents unrealistic predictions)

Example output:
```
Last recorded fuel price: 2.47 AED
Forecast for next month: 2.41 AED
Change: -0.06 AED
Expected decrease of 0.06 AED (2.3%)
```

##  Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib xgboost 
pip install scikit-optimize category_encoders yfinance joblib
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/Amaanish/Fuel-Prediction.git
cd Fuel-Prediction
```

2. **For Google Colab** (Recommended):
   - Upload the notebook file to Colab
   - Upload `FuelData.xlsx` to the Colab file system
   - Run all cells

3. **For Local Environment**:
   - Ensure all dependencies are installed
   - Place `FuelData.xlsx` in the same directory as the script
   - Run the Python script

### Usage
The model automatically:
1. Fetches real-time crude oil and gold prices
2. Loads and processes historical data
3. Trains the optimized XGBoost model
4. Generates next month's price forecast
5. Provides trend analysis and insights

## 📈 Model Improvements (V2.1 → V3)

| Aspect | V2.1 | V3 | Improvement |
|--------|------|----|-----------| 
| **Lag Features** | 3 months | 12 months | 4x more historical context |
| **Real-time Data** | None | Crude oil + Gold | Live market integration |
| **Hyperparameter Tuning** | 25 iterations | 50 iterations | 2x optimization depth |
| **Feature Count** | ~15 | 40+ | Richer feature space |
| **Prediction Stability** | Basic | Volatility-adjusted | Realistic forecasts |
| **Price Classification** | None | High/Normal/Low | Market context |

##  Future Enhancements

- **Extended forecast horizon**: 3-6 month predictions
- **Interactive dashboard**: Web-based visualization tool
- **Additional indicators**: USD/AED exchange rates, regional demand metrics
- **Ensemble methods**: Combining multiple algorithms
- **API deployment**: Real-time prediction service

##  Use Cases

- **Businesses**: Fleet management and budget planning
- **Consumers**: Personal fuel expense forecasting
- **Analysts**: Market trend analysis and reporting
- **Researchers**: Economic indicator correlation studies




---
*Last updated: August 13th 2025 | Model Version: V3.1*
