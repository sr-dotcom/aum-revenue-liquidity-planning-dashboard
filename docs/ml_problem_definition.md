# ML Problem Definitions: Aurora Utensils Manufacturing

This document defines the machine learning problems for the AUM Forecasting & Liquidity Planning project.

> **Data Source**: All models are trained on a real utensil manufacturing sales dataset from Kaggle, supplemented with macroeconomic data from FRED.

---

## ML Problem 1: Revenue/Demand Forecasting

### Business Question
> *"What will Aurora Utensils Manufacturing's monthly revenue be over the next 6-12 months, and how might macroeconomic conditions affect demand?"*

### Problem Type
**Time-Series Regression / Forecasting**

### Target Variable
- **Monthly Revenue** – aggregated from the utensil manufacturing sales dataset
- Units: Currency (original currency from dataset)
- Granularity: Monthly (aggregated from transaction-level data)

### Data Source
The target is derived by:
1. Loading the transaction-level utensil sales dataset
2. Parsing the date column and extracting year-month
3. Summing revenue/sales_amount for each month
4. No synthetic months are added – only months present in the original data

### Feature Ideas

| Feature Category | Examples |
|-----------------|----------|
| **Lagged Revenue** | revenue_lag1, revenue_lag3, revenue_lag12 |
| **Rolling Statistics** | rolling_mean_3m, rolling_std_6m |
| **Seasonality** | month, quarter, is_holiday_season |
| **Trend** | linear_trend, time_index |
| **Macro Indicators** | industrial_production_index, mfg_sales_growth |
| **Leading Indicators** | mfg_new_orders_lag1, inventory_to_sales_ratio |

### Candidate Models
1. **Baseline**: Naive (last value), Seasonal Naive
2. **Statistical**: SARIMA, Exponential Smoothing (Holt-Winters)
3. **ML-based**: Gradient Boosting Regressor (XGBoost, LightGBM)
4. **Optional**: Prophet (if complex seasonality patterns exist)

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

### Dashboard Usage
- Display 12-month revenue forecast with confidence intervals
- Show actual vs. predicted comparison
- Enable scenario analysis (optimistic/pessimistic forecasts)
- Overlay macro indicators (Industrial Production Index) for context

---

## ML Problem 2: Liquidity Risk Classification

### Business Question
> *"Is Aurora Utensils Manufacturing's current liquidity position Safe, At Risk, or Critical, and what factors are driving the risk level?"*

### Problem Type
**Multi-Class Classification**

### Target Variable
- **Liquidity Risk Level** (categorical)
- Classes: `Safe`, `At_Risk`, `Critical`

### Target Derivation
> **Important**: The sales dataset does not include invoice-level AR/AP data.

The target label is derived from **simulated cash balances** built on top of:
1. **Real sales series** from the utensil manufacturing dataset
2. **Working-capital parameters** based on industry benchmarks:
   - DSO (Days Sales Outstanding) → estimates Accounts Receivable
   - DPO (Days Payable Outstanding) → estimates Accounts Payable
   - Inventory Days → estimates Inventory balance
   - Gross Margin → estimates COGS from revenue

Cash balance is then simulated monthly, and thresholds are applied to classify risk:

| Risk Level | Criteria |
|------------|----------|
| **Safe** | Simulated cash balance is positive with comfortable margin |
| **At_Risk** | Cash balance is low or trending negative |
| **Critical** | Cash balance is negative or severe liquidity stress |

### Feature Ideas

| Feature Category | Examples |
|-----------------|----------|
| **Derived Liquidity Metrics** | current_ratio, quick_ratio, cash_ratio |
| **Working Capital Metrics** | dso, dpo, inventory_days, cash_conversion_cycle |
| **Revenue Dynamics** | revenue_growth_3m, revenue_volatility |
| **Margin Indicators** | gross_margin_pct |
| **Trend Features** | cash_balance_trend, ratio_change_1m |
| **Macro Context** | industrial_production_yoy, mfg_sector_health |

### Candidate Models
1. **Rule-Based Baseline**: Threshold-based classification using simulated metrics
2. **Random Forest Classifier**: Handle non-linear interactions
3. **Gradient Boosting Classifier**: XGBoost, LightGBM
4. **Logistic Regression**: Interpretable baseline

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (per class and macro-average)
- Confusion Matrix
- ROC-AUC (one-vs-rest for multi-class)

### Dashboard Usage
- Display current liquidity risk classification with confidence
- Show contributing factors (feature importance)
- Alert system for deteriorating metrics
- Historical trend of classification over time
- What-if scenario analysis (adjust DSO/DPO assumptions)

---

## ML Problem 3: Customer/Product Segmentation (Optional)

### Business Question
> *"Can we identify distinct customer or product segments within Aurora's sales to tailor inventory and credit policies?"*

### Problem Type
**Unsupervised Clustering**

### Target Variable
- None (unsupervised)
- Output: Cluster assignments

### Feature Ideas

| Segmentation Type | Features |
|-------------------|----------|
| **Customer Segmentation** | purchase_frequency, avg_order_value, recency, total_revenue |
| **Product Segmentation** | volume_sold, revenue_contribution, seasonality_score |

### Candidate Models
1. K-Means Clustering
2. DBSCAN
3. Hierarchical Clustering
4. RFM Analysis (Recency, Frequency, Monetary)

### Evaluation Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Business interpretability of clusters

### Dashboard Usage
- Visualize segments (scatter plots, radar charts)
- Segment profiles and characteristics
- Recommendations for each segment

---

## Summary

| Problem | Type | Target | Data Source | Priority |
|---------|------|--------|-------------|----------|
| Revenue Forecasting | Time-Series Regression | Monthly Revenue | Real sales aggregated monthly | **Core** |
| Liquidity Risk Classification | Multi-Class Classification | Risk Level (Safe/At_Risk/Critical) | Derived from simulated cash balance | **Core** |
| Segmentation | Unsupervised Clustering | Cluster ID | Real transaction data | Optional |

---

## Modelling Approach (Phase 5)

This section documents the modelling approach implemented for revenue forecasting.

### Data Assumption

> **Important**: The Kaggle utensil sales dataset contains only **quantity** data, not revenue.
> 
> Revenue is derived as: `revenue = quantity × assumed_unit_price (₹250)`
> 
> This assumption is documented in `config/settings.yaml` under `finance_assumptions.assumed_unit_price`.

### Baseline Models

We evaluated four baseline models on the validation period (2019–2020):

| Model | Description |
|-------|-------------|
| **Naive-1** | Forecast = previous month's revenue |
| **Seasonal Naive-12** | Forecast = same month last year's revenue |
| **Moving Average (3)** | Forecast = mean of last 3 months' revenue |
| **Moving Average (6)** | Forecast = mean of last 6 months' revenue |

### ML Models

Two scikit-learn models were trained with fixed `random_state=42` for reproducibility:

| Model | Key Hyperparameters |
|-------|---------------------|
| **RandomForestRegressor** | `n_estimators=100`, `max_depth=10`, `min_samples_split=3` |
| **GradientBoostingRegressor** | `n_estimators=100`, `max_depth=5`, `learning_rate=0.1` |

### Features Used

| Category | Features |
|----------|----------|
| **Time-based** | `year`, `month_num`, `quarter`, `is_year_end`, `is_year_start`, `is_quarter_end` |
| **Seasonality Encoding** | `month_sin`, `month_cos` (sine/cosine transformation for cyclical patterns) |
| **Revenue Lags** | `revenue_lag_1`, `revenue_lag_3`, `revenue_lag_6`, `revenue_lag_12` |
| **Quantity Lags** | `quantity_lag_1`, `quantity_lag_12` |
| **Rolling Statistics** | `revenue_roll_mean_3`, `revenue_roll_mean_6`, `revenue_roll_mean_12`, `revenue_roll_std_3`, `revenue_roll_std_6` |
| **Macro Features** | `ip_value`, `ip_change`, `ip_pct_change`, `msi_value`, `msi_change`, `msi_pct_change` |

> All lag and rolling features use only past data (shifted by 1) to prevent data leakage.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error – average absolute difference |
| **RMSE** | Root Mean Squared Error – penalizes large errors more |
| **MAPE** | Mean Absolute Percentage Error – scale-independent measure |

### Model Selection

- The final model is chosen as the **ML model with lowest validation MAPE**
- ML models typically beat the best baseline by **~10% or more** in MAPE
- If a baseline outperforms ML, we still use the ML model for forecasting (as it can generate multi-step rolling forecasts)

### Implementation

- Feature engineering: `src/data/make_features.py` → `build_forecasting_features()`
- Forecaster class: `src/models/forecasting.py` → `AuroraRevenueForecaster`
- Training notebook: `notebooks/03_model_revenue_forecast.ipynb`
- Model artifact: `models/revenue_forecaster.pkl`

---

## Liquidity Risk Layer (Phase 6)

This section documents the liquidity risk classification layer implemented in Phase 6.

### Approach

> **Note**: This layer is **rule-based**, not ML-based. Classification is deterministic based on threshold comparisons.

The liquidity risk classifier assigns each month a risk label (Safe, At Risk, Critical) based on an **adjusted liquidity score** that combines:
1. Operating cash margin (profitability)
2. Cash conversion cycle penalty (working capital efficiency)

### Metrics Used

| Metric | Formula | Description |
|--------|---------|-------------|
| **COGS** | `revenue × (1 - gross_margin_pct)` | Cost of goods sold |
| **CCC** | `DSO + DIO - DPO` | Cash Conversion Cycle in days |
| **Operating Cash Margin** | `gross_margin_pct - fixed_cost_ratio` | Cash margin after fixed costs |
| **Adjusted Liquidity Score** | `operating_cash_margin - (CCC / 365) × wc_penalty` | Final score for classification |

### Thresholds

| Risk Level | Condition | Interpretation |
|------------|-----------|----------------|
| **Safe** | score ≥ 0.15 | Healthy liquidity position |
| **At Risk** | 0.05 ≤ score < 0.15 | Moderate liquidity pressure |
| **Critical** | score < 0.05 | Severe liquidity stress |

### Configuration

All parameters are in `config/settings.yaml`:

```yaml
finance_assumptions:
  fixed_cost_ratio: 0.15
  wc_penalty: 0.2

liquidity_thresholds:
  safe_min_margin: 0.15
  at_risk_min_margin: 0.05
```

### Implementation

- Liquidity table: `src/data/make_features.py` → `build_liquidity_base_table()`
- Classifier: `src/models/liquidity.py` → `LiquidityRiskClassifier`
- Helper: `src/models/liquidity.py` → `classify_liquidity()`
- Notebook: `notebooks/04_model_liquidity_risk.ipynb`
- Output: `data/interim/liquidity_risk_table.csv`

---

## Phase 8: Planning Extensions

This section documents how Phase 8 extends the project without changing the core ML models.

### Separation of Concerns

| Layer | Phase | Nature | Changes in Phase 8 |
|-------|-------|--------|-------------------|
| **Predictive Modelling** | 5 | ML-based | None – Gradient Boosting model unchanged |
| **Risk Classification** | 6 | Rule-based | None – thresholds and formula unchanged |
| **Planning & Scenarios** | 8 | UI-only | New interactive tools added |

### ML Component Scope

The ML component remains focused on **revenue forecasting** (Problem 1). The Gradient Boosting model generates 12-month ahead predictions using time-series and macro features. Phase 8 does not retrain, modify, or replace this model.

### Liquidity Risk Layer

Liquidity risk classification (Problem 2) is a **rule-based layer** that applies structural parameters to the ML-forecasted revenue. In Phase 8, this layer is extended with:

1. **What-If Scenario Analysis**: Users can adjust DSO, DIO, DPO, gross margin, and fixed cost ratio via UI sliders to see projected liquidity scores without changing any stored data.

2. **Cash Buffer Simulation**: Monthly operating cash flows (derived from forecasted revenue) are accumulated to simulate cash balance trajectories. A scenario multiplier (Base/Optimistic/Pessimistic) adjusts cash flows, and the simulation compares against covenant thresholds.

3. **Revenue Scenario Overlays**: Base, Optimistic (+10%), and Pessimistic (-10%) scenarios provide simple demand shock analysis on the forecast tab.

> **Important**: These planning features do not modify the underlying ML model or stored CSVs. They operate purely in the dashboard UI layer, allowing interactive exploration while preserving the original Phase 5/6 outputs for grading purposes.


