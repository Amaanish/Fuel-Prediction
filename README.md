# UAE Fuel Price Prediction

Advanced machine learning project forecasting monthly fuel prices for Special 95 gasoline in the UAE using historical data and real-time economic indicators.

## Overview

This project combines historical fuel price data with live market indicators to predict UAE fuel prices 1-3 months ahead. Using XGBoost regression with optimized hyperparameters, the model achieves robust predictions by analyzing:

- Historical fuel price patterns (6-month lag features)
- Real-time crude oil prices (via Yahoo Finance API)
- Live gold prices scraped from UAE Gold Price
- Rolling statistics and volatility measures
- Advanced momentum indicators and ratio features
- Weighted training for recent data emphasis

## Key Features

- **Dataset Coverage**: January 2015 to present with monthly granularity
- **Real-time data integration**: Live crude oil prices via Yahoo Finance and gold prices via web scraping
- **Enhanced feature engineering**: Extended lag features with momentum indicators and volatility measures
- **Improved model stability**: Adaptive volatility-based smoothing with oil change detection
- **Automated data updates**: Standalone script for monthly data collection
- **Optimized hyperparameters**: Fine-tuned XGBoost parameters for time-series forecasting
- **Production-ready code**: Robust error handling with graceful fallbacks

## Technical Implementation

### Model Architecture

- **Algorithm**: XGBoost Regressor with L1 and L2 regularization
- **Validation**: Time Series Cross-Validation (adaptive splits based on dataset size)
- **Optimization**: Manually tuned hyperparameters with extensive testing
- **Features**: 25+ engineered features including extended lags, momentum, ratios, and moving averages
- **Weighting**: Exponential sample weights with stronger emphasis on recent data

### Model Performance

| Version | MAE (AED) | R² Score | Features | Prediction Method | Key Innovation |
|---------|-----------|----------|----------|-------------------|----------------|
| V2.1    | 0.1359    | 0.5377   | ~15      | Basic             | Initial model |
| V3.0    | 0.1360    | 0.5400   | 40+      | Volatility-adjusted | Real-time data |
| V3.2    | 0.1644    | 0.0895   | 40+      | Conservative dampening | Stability focus |
| **V3.3** | **0.1315** | **0.6270** | **26** | **Adaptive smoothing** | **Oil change detection** |

**Note**: V3.3 uses shallow trees (max_depth=1) for robust, interpretable predictions that respond accurately to oil price movements.

## Data Sources

- **Historical Data**: Custom compiled dataset (FuelData.xlsx)
- **Crude Oil**: Yahoo Finance ticker "CL=F" (WTI Crude Oil Futures)
- **Gold Prices**: UAE Gold Price website (24K per gram in AED)
- **UAE Fuel Prices**: Gulf News historical fuel rates (for monthly updates)

## Feature Engineering

The model incorporates comprehensive feature engineering for robust predictions:

```python
# Extended lag features (6-month window)
for lag in range(1, 7):
    df[f'fuel_lag_{lag}'] = df[FUEL_COL].shift(lag)
    df[f'oil_lag_{lag}'] = df[OIL_COL].shift(lag)
    df[f'gold_lag_{lag}'] = df[GOLD_COL].shift(lag)

# Momentum indicators
df['oil_momentum'] = df[OIL_COL].pct_change(periods=1)
df['fuel_momentum'] = df[FUEL_COL].pct_change(periods=1)

# Ratio and moving averages
df['gold_oil_ratio'] = df[GOLD_COL] / df[OIL_COL]
df['fuel_ma_3'] = df[FUEL_COL].rolling(window=3).mean()
df['oil_ma_3'] = df[OIL_COL].rolling(window=3).mean()
df['oil_fuel_diff'] = df[OIL_COL].pct_change() - df[FUEL_COL].pct_change()
df['oil_change_3m'] = (df[OIL_COL] - df[OIL_COL].shift(3)) / df[OIL_COL].shift(3)
```

### Weighted Training Strategy

```python
# Stronger exponential weights for recent data
weights = np.logspace(0.8, 2.2, num=len(y))
model.fit(X_train, y_train, sample_weight=weights[train_index])
```

## Prediction Methodology

### Adaptive Volatility Smoothing (V3.3)

The model implements intelligent prediction smoothing that adapts to both volatility and oil price changes:

```python
# Base volatility smoothing
volatility_factor = np.clip(oil_volatility / 18, 0.15, 0.45)

# Detect significant oil price drops
oil_change = (current_oil - last_oil) / last_oil
if oil_change < -0.10:  # 10% or more drop
    volatility_factor *= 0.5  # Highly aggressive prediction

smoothed_pred = ((1 - volatility_factor) * raw_pred) + (volatility_factor * current_fuel)
```

**How it works**:
- **Normal conditions**: Light smoothing (15-45% current price weight)
- **High volatility**: More conservative approach
- **Large oil drops (>10%)**: Halved smoothing for highly responsive predictions
- Captures both gradual trends and sharp market movements

### Output Format

```
=== Model Performance Stats ===
MAE: 0.0950
R² Score: 0.7500

=== Next Month Fuel Forecast ===
Current Oil: $57.84
Last Fuel: 2.58 AED
Predicted Fuel for Next Month: 2.42 AED
Predicted Change: -0.16 AED

Expected decrease of 0.16 AED
```

## Installation

### Dependencies

Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost
pip install yfinance beautifulsoup4 requests openpyxl
```

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/Amaanish/Fuel-Prediction.git
cd Fuel-Prediction
```

2. **For Google Colab (Recommended):**
   - Upload the notebook file (`Fuel_Predict_XGBOOST_3.3.ipynb`) to Colab
   - Upload `FuelData.xlsx` to the Colab file system
   - Run all cells

3. **For Local Environment:**
   - Ensure all dependencies are installed
   - Place `FuelData.xlsx` in the same directory as the script
   - Run: `python fuel_predict_v33.py`

## Usage

### Running the Prediction Model

The model automatically:

1. Fetches real-time crude oil prices from Yahoo Finance
2. Scrapes current gold prices from UAE Gold Price website
3. Loads and processes historical data from Excel
4. Engineers 25+ time-series features
5. Trains the XGBoost model with weighted samples
6. Generates next month's price forecast with adaptive smoothing
7. Provides performance metrics and trend analysis

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

## Model Hyperparameters (V3.3)

```python
XGBRegressor(
    n_estimators=900,         # More trees for better learning
    max_depth=1,              # Shallow trees for simple, robust patterns
    learning_rate=0.015,      # Balanced learning rate
    subsample=0.75,           # 75% row sampling
    colsample_bytree=0.75,    # 75% feature sampling
    reg_lambda=30,            # L2 regularization
    reg_alpha=15,             # L1 regularization
    random_state=8,
    base_score=recent_mean    # Recent 6-month average
)
```

## Version Evolution

| Aspect | V2.1 | V3.0 | V3.2 | **V3.3** |
|--------|------|------|------|----------|
| Lag Features | 3 months | 12 months | 12 months | 6 months |
| Real-time Data | None | Oil + Gold | Oil + Gold | Oil + Gold |
| Feature Count | ~15 | 40+ | 40+ | 26 |
| Prediction Method | Basic | Volatility-adj | Conservative | Adaptive |
| MAE (AED) | 0.1359 | 0.1360 | 0.1644 | **0.1315** |
| R² Score | 0.5377 | 0.5400 | 0.0895 | **0.6270** |
| Sample Weighting | No | No | No | Yes (Strong) |
| Oil Change Detection | No | No | No | **Yes** |
| Error Handling | Basic | Moderate | Advanced | **Production** |

## What's New in V3.3

 **Major Improvements:**

1. **Enhanced Feature Engineering**
   - Extended to 6-month lag features for fuel, oil, and gold
   - Added fuel momentum tracking
   - Included 3-month moving averages
   - Oil-fuel price difference indicator
   - 3-month oil change tracking

2. **Adaptive Smoothing**
   - Detects significant oil price drops (>10%)
   - Halves smoothing factor for highly responsive predictions
   - More responsive to large market movements

3. **Improved Model Architecture**
   - Increased to 900 estimators with shallow trees (max_depth=1)
   - Balanced learning rate (0.015)
   - Stronger L1 regularization (reg_alpha=15)
   - Enhanced sample weighting (logspace 0.8 to 2.2)
   - Base score from recent 6-month average

4. **Better Performance**
   - MAE of 0.1315 AED (~5.1% average error)
   - R² Score of 0.6270 (62.7% variance explained)
   - Highly accurate predictions during oil price volatility
   - Shallow trees prevent overfitting for robust predictions

5. **Production-Ready Code**
   - Comprehensive error handling with graceful fallbacks
   - Sanity checks and timeouts for external API calls
   - Clean, minimal code structure

## Technical Deep Dive

### Sample Weighting

```python
weights = np.logspace(0.8, 2.2, num=len(y))
# Recent data gets ~25x more weight than older data
# More aggressive than previous version (0.5 to 2.0)
```

### Adaptive Smoothing Logic

| Condition | Volatility Factor | Prediction Behavior |
|-----------|------------------|---------------------|
| Low volatility | 0.15-0.25 | Trust model (75-85% weight) |
| Medium volatility | 0.25-0.45 | Balanced approach |
| High volatility | 0.45 | Conservative (55% weight) |
| Oil drop >10% | Factor × 0.5 | Highly aggressive |

### Time Series Cross-Validation

```python
tscv = TimeSeriesSplit(n_splits=min(3, len(X)-1))
# Adaptive splits for datasets of varying sizes
```

## Performance Metrics

- **MAE**: 0.1315 AED (~5.1% average error)
- **R² Score**: 0.6270 (62.7% variance explained)
- **Interpretation**: Predictions typically within 13 fils of actual price
- **Accuracy**: Correctly predicted 2.42 AED when oil dropped to $57.78

## Applications

- **Fleet Management**: Budget planning for transportation companies
- **Consumer Planning**: Personal fuel expense forecasting
- **Market Analysis**: Trend analysis for energy sector reports
- **Economic Research**: Correlation studies with regional indicators
- **Government Policy**: Price stabilization strategy support

## Project Structure

```
Fuel-Prediction/
├── FuelData.xlsx                      # Historical dataset (2015-present)
├── Fuel_Predict_XGBOOST_3.3.ipynb    # Main prediction model
├── update.py                          # Automated data collection script
└── README.md                          # This documentation
```

## Known Limitations

- Monthly granularity only (cannot predict weekly/daily fluctuations)
- Web scraping dependent on external website structure
- Historical data limited to UAE Special 95 gasoline
- Model assumes historical patterns continue
- Gold price scraping may occasionally fail (uses fallback)

## Future Enhancements

- [ ] Multi-month forecasting (3-6 months ahead)
- [ ] Interactive web dashboard
- [ ] Additional economic indicators (USD/AED exchange rate)
- [ ] Ensemble methods (XGBoost + LSTM)
- [ ] REST API deployment
- [ ] Automated monthly retraining

## Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering ideas
- Alternative machine learning algorithms
- Enhanced data collection methods
- Visualization tools


## Contact & Support

- **Repository**: [github.com/Amaanish/Fuel-Prediction](https://github.com/Amaanish/Fuel-Prediction)
- **Issues**: Report bugs or request features via GitHub Issues
- **Author**: Amaan

---

**Last Updated**: December 2024 | **Current Version**: V3.3 
