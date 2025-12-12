# Project Charter: Aurora Utensils Manufacturing – Forecasting & Liquidity Planning

## Background and Motivation

Manufacturing companies, particularly those in the consumer durables sector, face significant challenges in managing working capital and maintaining liquidity. For a mid-size utensil manufacturer like **Aurora Utensils Manufacturing (AUM)**, these challenges are amplified by:

- **Seasonal demand patterns** – Kitchen and home products often experience demand spikes around holidays and wedding seasons
- **Long production cycles** – Manufacturing lead times create timing mismatches between cash outflows (raw materials, labor) and inflows (customer payments)
- **Thin margins** – Competitive pricing pressure in the utensils market requires efficient cash management
- **Supply chain dependencies** – Raw material costs (stainless steel) are sensitive to commodity price fluctuations

Poor liquidity planning can result in missed growth opportunities, supplier relationship damage, or in severe cases, operational disruptions. This project aims to provide AUM's finance team with data-driven tools to anticipate revenue fluctuations and proactively manage liquidity risk.

---

## Data Foundation

This project uses a **real utensil manufacturing sales dataset** from Kaggle as the core transaction history for Aurora Utensils Manufacturing. The dataset contains several years of transaction-level sales data including dates, products, regions/customers, quantities, and revenue amounts.

We supplement this with macroeconomic and industry data from FRED and U.S. Census sources to provide context for forecasting and scenario analysis.

---

## Project Objectives

1. **Build accurate revenue forecasting models** that predict monthly/quarterly sales using historical transaction data and macroeconomic indicators

2. **Simulate working capital dynamics** by deriving realistic liquidity metrics (DSO, DPO, CCC) using industry-standard financial ratios

3. **Develop a liquidity risk classification system** that categorizes the company's liquidity health as Safe, At Risk, or Critical

4. **Create an interactive Streamlit dashboard** that enables the finance team to explore forecasts, monitor liquidity metrics, and run scenario analyses

5. **Document methodology and assumptions transparently** to ensure the models can be validated, updated, and trusted by stakeholders

---

## Scope

### In Scope
- Revenue/demand forecasting using time-series and ML regression models
- Derivation of working capital metrics from real revenue data and realistic ratio assumptions
- Multi-class classification for liquidity risk assessment
- Exploratory data analysis of sales patterns and macro correlations
- Interactive dashboard for visualization and scenario planning
- Documentation of data sources, methodology, and model performance

### Out of Scope
- Real-time data integration or production deployment
- Integration with AUM's ERP or accounting systems
- Inventory optimization or supply chain modeling
- Credit scoring or customer-level risk assessment
- Multi-currency or international operations modeling

---

## Final Deliverables

| Deliverable | Description |
|-------------|-------------|
| **ML Models** | Trained revenue forecasting and liquidity risk classification models |
| **Streamlit Dashboard** | Interactive app for exploring forecasts, liquidity metrics, and scenarios |
| **Technical Report** | Documentation of methodology, data sources, model evaluation, and limitations |
| **Jupyter Notebooks** | Reproducible EDA and modeling workflows |
| **Codebase** | Clean, modular Python code organized for maintainability |

---

## Assumptions and Limitations

### Data Assumptions
1. We use a **real utensil manufacturing sales dataset** from Kaggle as Aurora's historical transaction data
2. The dataset represents transaction-level sales (date, product, region/customer, quantity, revenue)
3. We supplement with real macroeconomic data from FRED (Industrial Production Index, Manufacturers' Sales & Inventories)

### Working Capital Derivation
1. **We do NOT have invoice-level AR/AP data** from the sales dataset
2. Working-capital and liquidity metrics (Accounts Receivable, Accounts Payable, Inventory, Cash) are **derived by applying realistic financial ratios** from industry data to the real sales history
3. Initial ratio assumptions (DSO ~45 days, DPO ~40 days, Inventory Days ~60, Gross Margin ~30%) are based on manufacturing industry benchmarks and can be refined using SEC financial statement data

### No Synthetic Data Rule
> **Critical**: We follow a strict "no synthetic transactional data" rule:
> - All base transactional records come from real public datasets
> - We may clean, aggregate, join, and derive new columns
> - We do NOT fabricate new orders, customers, or time periods not present in the original data
> - Derived metrics use transparent formulas applied to real-world ratios

### Limitations
1. Liquidity simulation is based on assumed ratios, not actual AR/AP aging data
2. Forecasts are trained on historical patterns and may not capture unprecedented events
3. Industry benchmarks may not perfectly match Aurora's specific business model

---

## Phase 5 – Revenue Forecasting (Summary)

This section summarizes the Phase 5 implementation of revenue forecasting.

- **Target**: Monthly revenue of Aurora Utensils Manufacturing (AUM), derived from quantity × assumed unit price (₹250)
- **Horizon**: 12-month ahead forecast (rolling, starting from end of historical data)
- **Method**: Gradient Boosting (or Random Forest) on engineered time-series and macro features
  - Time-based features: year, month, quarter, is_year_end, is_quarter_end
  - Seasonality: sine/cosine encoding of month
  - Lag features: revenue_lag_1, revenue_lag_3, revenue_lag_6, revenue_lag_12
  - Rolling statistics: 3/6/12-month rolling means and standard deviations
  - Macro indicators: Industrial Production Index (IPMAN), Manufacturers' Shipments (AMTMVS)
- **Validation**: 
  - Train: 2016–2018 (36 months, minus initial lag window)
  - Validation: 2019–2020 (24 months)
  - Comparison against Naive-1, Seasonal Naive-12, and Moving Average baselines
- **Key Outputs**:
  - Forecast CSV: `data/interim/revenue_forecast_12m.csv`
  - Trained model: `models/revenue_forecaster.pkl`
  - Feature list: `models/feature_list.txt`

---

## Phase 6 – Liquidity Planning (Summary)

This section summarizes the Phase 6 implementation of liquidity risk classification.

- **Objective**: Classify each month's liquidity position as Safe, At Risk, or Critical
- **Inputs**:
  - Historical revenue: `data/interim/monthly_sales.csv`
  - Revenue forecast: `data/interim/revenue_forecast_12m.csv`
  - Working capital assumptions from `config/settings.yaml`
- **Method**: Rule-based classification using adjusted liquidity score
  - Computes Cash Conversion Cycle (CCC = DSO + DIO - DPO)
  - Calculates operating cash margin (gross margin - fixed costs)
  - Applies working capital penalty to derive adjusted score
  - Classifies: Safe (≥0.15), At Risk (0.05-0.15), Critical (<0.05)
- **Key Outputs**:
  - Liquidity base table: `data/interim/liquidity_base_table.csv`
  - Risk classification table: `data/interim/liquidity_risk_table.csv`
  - Analysis notebook: `notebooks/04_model_liquidity_risk.ipynb`

---

## Phase 7 – Dashboard Integration (Summary)

This section summarizes the Phase 7 implementation of the Streamlit dashboard.

- **Objective**: Integrate revenue forecasts and liquidity risk into an interactive dashboard
- **Framework**: Streamlit with Plotly visualizations
- **Structure**:
  - 🏠 Overview: Key metrics and combined historical + forecast chart
  - 📈 Revenue Forecast: Forecast details, YoY comparison, CSV download
  - 🎯 Liquidity Risk: Structural snapshot, cash flow timeline, risk table
- **Key Outputs**:
  - Data loader: `app/data_loader.py`
  - Dashboard: `app/dashboard.py`
- **Run Command**: `streamlit run app/dashboard.py`

---

## Phase 8 – Planning Enhancements (Summary)

This section summarizes the Phase 8 implementation of planning-grade features.

- **Objective**: Transform the dashboard from a reporting tool into an interactive planning cockpit
- **What-If Liquidity Planner**:
  - Individual sliders for DSO, DIO, DPO (days), Gross Margin (%), and Fixed Cost Ratio (%)
  - Computes CCC, operating margin, and adjusted liquidity score live
  - Displays projected risk band (Safe / At Risk / Critical) based on config thresholds
- **Cash Buffer & Covenant Simulation**:
  - User inputs for starting cash (default ₹50 Cr) and minimum covenant (₹25 Cr)
  - Scenario selector: Base / Optimistic (+10%) / Pessimistic (-10%) multiplier on cash flows
  - Plots cumulative cash balance trajectory over 72 months
  - Highlights covenant breaches with red markers and summary message
- **Revenue Scenarios**:
  - Base / Optimistic (+10%) / Pessimistic (-10%) selector on Revenue Forecast tab
  - Adjusts only forecast months; historical data unchanged
  - Overlays scenario forecast on chart with distinct colours
- **Key Constraint**: All planning calculations are UI-side only; no CSV or model changes

### Benefits

Phase 8 turns the project from **analytics + reporting** into **interactive planning**:
- Finance teams can test "what-if" scenarios without modifying data
- Stress-testing cash flow projections becomes instant and visual
- Stakeholders can explore upside/downside ranges in live conversations


