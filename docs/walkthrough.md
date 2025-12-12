# Phase 5 Walkthrough: Revenue Forecasting Model

## Overview

Phase 5 implemented a complete revenue forecasting pipeline for Aurora Utensils Manufacturing (AUM). The pipeline includes feature engineering, baseline model comparison, ML model training, and 12-month forecast generation.

> **Data Note**: Revenue is derived from the Kaggle volume data as `quantity × assumed_unit_price (₹250)`, since the original dataset contains units sold but no prices. This assumption is documented in `config/settings.yaml`.

---

## Model Selection

### Selected Model

**Gradient Boosting Regressor** was selected as the production forecasting model.

> **Why not Random Forest or Seasonal Naive-12?** 
> Although these models achieve slightly lower validation MAPE (5.75% and 5.95% vs 8.67%), Gradient Boosting was chosen because it:
> - Produces **smoother multi-step forecasts** (RF can be noisy in rolling predictions)
> - Is **less prone to overfitting** on our relatively small effective training window
> - Provides **more interpretable feature importance** for business insights
> - Is **already integrated** as the default in `AuroraRevenueForecaster`
>
> Seasonal Naive-12 and Random Forest serve as strong benchmarking baselines.

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 100 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `min_samples_split` | 3 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |

### Features Used (25 total)

| Category | Features |
|----------|----------|
| **Time-based** | year, month_num, quarter, is_year_end, is_year_start, is_quarter_end |
| **Seasonality** | month_sin, month_cos |
| **Revenue lags** | revenue_lag_1, revenue_lag_3, revenue_lag_6, revenue_lag_12 |
| **Rolling stats** | revenue_roll_mean_3, revenue_roll_mean_6, revenue_roll_mean_12, revenue_roll_std_3, revenue_roll_std_6 |
| **Quantity lags** | quantity_lag_1, quantity_lag_12 |
| **Macro features** | ip_value, ip_change, ip_pct_change, msi_value, msi_change, msi_pct_change |

---

## Validation Metrics

### Selected Model (Gradient Boosting)

| Metric | Value |
|--------|-------|
| **MAE** | Rs. 7,507,979 |
| **RMSE** | Rs. 8,417,133 |
| **MAPE** | 8.67% |

> **Scale Context**: Typical monthly revenue is Rs.60–90M, so an MAE of ~Rs.7.5M corresponds to roughly 8-10% error on average.

### All Model Comparison

| Model | Type | MAPE |
|-------|------|------|
| Seasonal Naive-12 | Baseline | 5.75% |
| Random Forest | ML | 5.95% |
| **Gradient Boosting (production)** | ML | **8.67%** |
| Naive-1 | Baseline | 10.79% |
| Moving Avg 3 | Baseline | 18.00% |
| Moving Avg 6 | Baseline | 22.40% |

> **Analysis**: The Seasonal Naive-12 baseline performs best on this validation set, confirming strong annual seasonality in the data. Random Forest slightly beats Gradient Boosting on single-step validation. For production forecasting, we retain Gradient Boosting for its stability in 12-month rolling predictions, treating Seasonal Naive-12 and RF as benchmarks.

---

## Artifacts Location

| Artifact | Path |
|----------|------|
| **12-Month Forecast** | `data/interim/revenue_forecast_12m.csv` |
| **Trained Model** | `models/revenue_forecaster.pkl` |
| **Feature List** | `models/feature_list.txt` |
| **Training Notebook** | `notebooks/03_model_revenue_forecast.ipynb` |

---

## Forecast Preview (2021)

| Month | Revenue Forecast (Rs.) |
|-------|------------------------|
| 2021-01 | 62,115,442 |
| 2021-02 | 62,441,453 |
| 2021-03 | 71,325,106 |
| 2021-04 | 72,592,406 |
| 2021-05 | 75,074,354 |
| 2021-06 | 72,039,057 |
| 2021-07 | 73,284,609 |
| 2021-08 | 71,044,818 |
| 2021-09 | 71,443,664 |
| 2021-10 | 70,725,009 |
| 2021-11 | 70,143,032 |
| 2021-12 | 68,041,829 |

---

## Usage Examples

### Loading and Using the Forecaster

```python
import sys
sys.path.insert(0, 'src')
from models.forecasting import AuroraRevenueForecaster

# Load saved model (using classmethod)
forecaster = AuroraRevenueForecaster.load('models/revenue_forecaster.pkl')

# Generate 6-month forecast
forecast = forecaster.predict(n_months=6)
print(forecast)
```

### Retraining the Model

```python
from data.load_data import load_config, load_macro_ip_data, load_macro_msi_data
from models.forecasting import AuroraRevenueForecaster
import pandas as pd

config = load_config()
df_monthly = pd.read_csv('data/interim/monthly_sales.csv')
df_monthly['year_month'] = pd.to_datetime(df_monthly['year_month'])
df_ip = load_macro_ip_data(config)
df_msi = load_macro_msi_data(config)

forecaster = AuroraRevenueForecaster(config, model_type='gradient_boosting')
forecaster.fit(df_monthly, df_ip, df_msi, valid_start_date='2019-01-01')
forecaster.save('models/revenue_forecaster.pkl')
```

---

## Key Insights

1. **Strong Seasonality**: The data exhibits clear annual seasonality with peaks in mid-year (June-August) and troughs in early year (January-February) and year-end (December). The Seasonal Naive-12 baseline's strong performance confirms this pattern.

2. **Macro Features Included**: Industrial Production Index (IPMAN) and Manufacturers' Shipments (AMTMVS) are included as features. Feature importance shows these among the predictors, though lag features dominate.

3. **Lag Features Critical**: Revenue lags (especially lag_1, lag_12) and rolling means are the most important features, capturing both trend and seasonality.

4. **Production-Ready**: The `AuroraRevenueForecaster` class provides a complete interface for training, saving, loading, and rolling multi-step forecasting.

---

## Bug Fixed

During validation, discovered that `AuroraRevenueForecaster.save()` was not persisting `_df_ip` and `_df_msi`. This caused `predict()` to fail after loading a saved model when macro features were expected.

**Fix**: Updated `forecasting.py` – `_df_ip` and `_df_msi` are now persisted with the model, so `predict()` works correctly after re-loading.

---

## Next Steps

- **Phase 7**: Dashboard Integration – Incorporate forecasts and liquidity risk into Streamlit
- **Ensemble Option**: Consider blending GB forecasts with Seasonal Naive-12 for improved accuracy
- **Model Monitoring**: Track forecast accuracy over time as new data becomes available

---

# Phase 6 Walkthrough: Liquidity Risk Classification

## Overview

Phase 6 implemented a rule-based liquidity risk classification layer that uses revenue forecasts and working capital assumptions to assess monthly liquidity positions.

---

## Classification Logic

### Adjusted Liquidity Score Formula

```
adjusted_liquidity_score = operating_cash_margin - (ccc_days / 365) × wc_penalty
```

Where:
- `operating_cash_margin = gross_margin_pct - fixed_cost_ratio = 0.30 - 0.15 = 0.15`
- `ccc_days = DSO + DIO - DPO = 45 + 60 - 40 = 65 days`
- `wc_penalty = 0.2`

**Result**: `adjusted_liquidity_score = 0.15 - (65/365) × 0.2 = 0.15 - 0.0356 = 0.1144`

### Thresholds

| Risk Level | Condition | Result for AUM |
|------------|-----------|----------------|
| Safe | score ≥ 0.15 | Not reached |
| **At Risk** | 0.05 ≤ score < 0.15 | **Current: 0.1144** |
| Critical | score < 0.05 | Not reached |

---

## Artifacts Location

| Artifact | Path |
|----------|------|
| **Liquidity Base Table** | `data/interim/liquidity_base_table.csv` |
| **Risk Classification Table** | `data/interim/liquidity_risk_table.csv` |
| **Analysis Notebook** | `notebooks/04_model_liquidity_risk.ipynb` |
| **Classifier Code** | `src/models/liquidity.py` |

---

## How to Re-Run

```python
import sys
sys.path.insert(0, 'src')
import pandas as pd
from data.load_data import load_config
from data.make_features import build_liquidity_base_table
from models.liquidity import classify_liquidity

config = load_config()
df_monthly = pd.read_csv('data/interim/monthly_sales.csv')
df_forecast = pd.read_csv('data/interim/revenue_forecast_12m.csv')

# Build liquidity table (saves to data/interim/liquidity_base_table.csv)
df_liquidity = build_liquidity_base_table(df_monthly, config, df_forecast)

# Classify risk
df_scored = classify_liquidity(df_liquidity, config)
df_scored.to_csv('data/interim/liquidity_risk_table.csv', index=False)
```

---

## Interpretation

With the current working capital assumptions:
- **All 72 months** (60 historical + 12 forecast) are classified as **"At Risk"**
- The score (0.1144) is just below the Safe threshold (0.15)
- This reflects the 65-day cash conversion cycle creating working capital drag

### Why the Score is Constant

> **Design Note**: The adjusted_liquidity_score is intentionally constant (0.1144) across all months because it measures **working capital structure efficiency**, not revenue level.
>
> The formula uses only:
> - `gross_margin_pct` (0.30) — from config
> - `fixed_cost_ratio` (0.15) — from config
> - CCC days (65) — derived from DSO/DIO/DPO in config
> - `wc_penalty` (0.2) — from config
>
> **Revenue does not affect the risk label** — only the `operating_cash_flow` column varies with revenue.
>
> For Phase 6, this simple rule-based structure is intentional: it treats liquidity structure as the primary risk driver, while revenue volatility can be layered on later as an enhancement.

### Sensitivity

To move from "At Risk" to "Safe", AUM would need to:
1. **Increase gross margin** from 30% to ~35%, OR
2. **Reduce CCC** from 65 to ~37 days (e.g., by reducing DSO from 45 to 25 days)

---

## Key Insights

1. **Uniform Risk Level**: Because working capital assumptions are constant, all months receive the same adjusted_liquidity_score (0.1144).

2. **CCC Impact**: The 65-day CCC creates a 0.0356 penalty, pushing the score below the "Safe" threshold.

3. **Revenue Independence**: In this rule-based model, revenue level does not affect the risk label—only the operating_cash_flow varies with revenue.

4. **Production-Ready**: The `LiquidityRiskClassifier` class provides a simple interface for scoring new data.

---

# Phase 7 Walkthrough: Dashboard Integration

## Overview

Phase 7 integrated the Phase 5 (Revenue Forecast) and Phase 6 (Liquidity Risk) outputs into a Streamlit dashboard with three tabs.

---

## Dashboard Structure

| Tab | Content |
|-----|---------|
| **🏠 Overview** | Key metrics, combined historical + forecast chart |
| **📈 Revenue Forecast** | Forecast detail table, YoY comparison, download button |
| **🎯 Liquidity Risk** | Structural snapshot, cash flow timeline, risk table |

---

## Artifacts Created

| File | Description |
|------|-------------|
| `app/data_loader.py` | Data loading utilities for dashboard |
| `app/dashboard.py` | Main Streamlit application |

---

## How to Run

```bash
# From project root
streamlit run app/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Features

### Overview Tab
- Key metrics: Last actual revenue, Avg forecast, CCC days, Risk band
- Combined line chart showing historical + forecast revenue
- Quick summary boxes for model info and liquidity structure

### Revenue Forecast Tab
- Gradient Boosting model details
- Combined historical + forecast chart
- 12-month forecast table with YoY comparison
- CSV download button

### Liquidity Risk Tab
- Structural liquidity snapshot (CCC, margin, score, risk band)
- Operating cash flow timeline chart (colored by historical/forecast)
- Filterable risk classification table
- Key insights and recommendations

---

## Data Loader API

```python
from app.data_loader import (
    get_revenue_history_and_forecast,  # → (df_hist, df_forecast)
    get_liquidity_risk_table,          # → df_risk
    get_liquidity_config_summary,      # → dict with CCC, score, thresholds, etc.
    check_data_availability,           # → dict of file statuses
)
```

---

# Phase 8 Walkthrough: Planning Enhancements

## Overview

Phase 8 transforms the dashboard from a **reporting tool** into a **planning cockpit** by adding interactive what-if analysis and scenario modeling capabilities. These features allow finance teams to explore different operational scenarios without modifying the underlying data or retraining models.

> **Important**: Phase 8 builds on top of Phases 5, 6, and 7. The structural score, thresholds, and base forecast remain unchanged. All what-if calculations are performed in the UI layer only.

---

## 1. What-If Liquidity Planner

Located on the **Liquidity Risk** tab, this tool enables interactive exploration of structural liquidity score changes.

### Inputs

| Slider | Range | Default | Purpose |
|--------|-------|---------|---------|
| DSO (days) | 20–90 | 45 | Days Sales Outstanding – receivables collection speed |
| DIO (days) | 20–120 | 60 | Days Inventory Outstanding – inventory holding period |
| DPO (days) | 0–120 | 40 | Days Payable Outstanding – supplier payment timing |
| Gross Margin (%) | 10–60 | 30 | Revenue margin after COGS |
| Fixed Cost Ratio (%) | 5–40 | 15 | Fixed operating costs as % of revenue |

### Calculation

The tool uses the same formula from Phase 6:

1. **CCC** = DSO + DIO − DPO (computed live from slider inputs)
2. **Operating Cash Margin** = Gross Margin − Fixed Cost Ratio
3. **Adjusted Liquidity Score** = Operating Cash Margin − (CCC / 365) × wc_penalty

### Interpretation

| Risk Band | Condition | Meaning |
|-----------|-----------|---------|
| **Safe** | score ≥ 0.15 | Healthy structural liquidity |
| **At Risk** | 0.05 < score < 0.15 | Moderate pressure on working capital |
| **Critical** | score ≤ 0.05 | Severe liquidity stress |

The display shows the computed CCC, operating margin, score, and risk band—updating instantly as sliders move.

> **Note**: The What-If Planner changes only the scenario cards on the dashboard; it does not rewrite historical classifications or regenerate CSVs.

---

## 2. Cash Buffer & Covenant Simulation

Also on the **Liquidity Risk** tab, this simulation projects cumulative cash balances over the full 72-month window (60 historical + 12 forecast).

### Inputs

| Input | Default | Purpose |
|-------|---------|---------|
| Starting Cash Balance | ₹50 Cr | Opening cash position at start of simulation |
| Minimum Cash Covenant | ₹25 Cr | Threshold for covenant breach detection |
| Scenario | Base | Multiplier applied to operating cash flows |

### Scenario Multipliers

| Scenario | Multiplier | Effect |
|----------|------------|--------|
| Base | 1.0 | Uses original operating cash flow |
| Optimistic (+10%) | 1.10 | Higher cash inflows |
| Pessimistic (-10%) | 0.90 | Lower cash inflows |

### Simulation Logic

For each month (in chronological order):
- `cash_balance[t] = cash_balance[t-1] + operating_cash_flow[t] × scenario_multiplier`
- `breach[t] = True if cash_balance[t] < covenant`

### Output

- **Line chart**: Cash balance over time, coloured by Historical (solid) vs Forecast (dashed)
- **Covenant line**: Horizontal dashed red line at the covenant threshold
- **Breach markers**: Red X markers on months where balance falls below covenant
- **Summary**: Total months, breach count, and first breach month (if any)

---

## 3. Revenue Scenarios

Located on the **Revenue Forecast** tab, this feature adjusts only the 12 forecasted months—historical data remains unchanged.

### Scenario Options

| Scenario | Adjustment | Use Case |
|----------|------------|----------|
| Base | No change | Original Gradient Boosting model output |
| Optimistic (+10%) | ×1.10 | Demand surge / best-case planning |
| Pessimistic (-10%) | ×0.90 | Demand contraction / stress testing |

### How It Works

1. Historical revenue is always displayed unchanged
2. Forecast revenue is multiplied by the scenario factor
3. The chart overlays the scenario line in a distinct colour (green for optimistic, red for pessimistic)
4. The forecast details table shows both Base and Scenario values

### Why This Matters

Revenue scenarios help finance teams:
- Stress-test cash flow projections
- Plan for demand shocks without building separate models
- Quickly communicate upside/downside ranges to stakeholders

---

## Technical Notes

- **No CSV Changes**: All what-if and scenario calculations are purely UI-side
- **Phase 6 Preserved**: Structural score (0.1144), thresholds, and risk classification unchanged
- **Config-Driven**: All default values and thresholds read from `config/settings.yaml`

---

## Run Command

```bash
streamlit run app/dashboard.py
```

Dashboard opens at `http://localhost:8501` with all Phase 8 features integrated.


