# Aurora Utensils Manufacturing (AUM) – Forecasting & Liquidity Planning

## Project Overview

This project simulates **Aurora Utensils Manufacturing (AUM)**, a mid-size stainless-steel utensils and kitchenware manufacturer, to build a comprehensive **forecasting and liquidity planning system**. We use a real utensil manufacturing sales dataset from Kaggle as Aurora's transaction history, supplemented with macroeconomic data from FRED and U.S. Census sources.

## About Aurora Utensils Manufacturing (AUM)

Aurora Utensils Manufacturing is a fictional mid-size company operating in the consumer durables manufacturing sector. AUM produces stainless-steel utensils and kitchenware products, selling both domestically and internationally. Like many manufacturing firms, AUM faces challenges with:

- Seasonal demand fluctuations  
- Working capital management  
- Cash flow timing mismatches between receivables and payables  
- Sensitivity to macroeconomic cycles  

## Project Objectives

- **Revenue / Demand Forecasting** – Build ML models to predict future sales and demand patterns  
- **Liquidity Simulation** – Derive working capital metrics (DSO, DIO, DPO, cash conversion cycle) using realistic financial ratios  
- **Liquidity Risk Classification** – Classify liquidity status as **Safe / At Risk / Critical**  
- **Interactive Dashboard** – Deliver a Streamlit-based decision support tool for financial planning  

## ML Problems

| Problem                         | Type                     | Description                                                                 |
|---------------------------------|--------------------------|-----------------------------------------------------------------------------|
| Revenue Forecasting             | Time-Series Regression   | Predict monthly/quarterly revenue using historical sales and macro indicators |
| Liquidity Risk Classification   | Multi-Class Classification | Classify liquidity health into Safe / At Risk / Critical categories          |
| Customer/Product Segmentation (Optional) | Unsupervised Clustering | Segment customers or products for targeted analysis                          |

## Phase 8 Planning Features

| Feature                         | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| What-If Liquidity Planner       | Adjust DSO, DIO, DPO, gross margin, and fixed costs; see live CCC and risk band |
| Cash Buffer & Covenant Simulation | Simulate 72-month cash balance trajectory with scenario multipliers         |
| Revenue Scenarios               | Apply Base / Optimistic (+10%) / Pessimistic (−10%) to forecasted months    |

> **Note:** Phase 8 is a pure planning/UI layer – Phase 5/6 data artifacts remain unchanged for grading and reproducibility.

---

## Data Policy

**Critical:** This project follows strict data integrity rules.

### Raw Data Sources

- **Core Sales Data:** Real utensil manufacturing sales dataset from Kaggle (transaction-level)  
- **Macro Data:** Industrial Production Index and Manufacturers' Sales & Inventories from FRED  

### What We DO

- ✅ Use only real public datasets as raw inputs  
- ✅ Clean, aggregate, and join tables from real sources  
- ✅ Derive new columns using transparent formulas (e.g., DSO, DPO, cash balance)  
- ✅ Apply industry-standard financial ratios to simulate working capital  

### What We DO NOT Do

- ❌ Fabricate new orders, customers, or time periods  
- ❌ Generate synthetic transactional data  
- ❌ Invent AR/AP aging data without clear derivation methodology  

### Liquidity Variable Derivation

Since the sales dataset does not include invoice-level AR/AP data, liquidity variables are derived using:

- **DSO (Days Sales Outstanding):** Applied to revenue to estimate Accounts Receivable  
- **DPO (Days Payable Outstanding):** Applied to COGS to estimate Accounts Payable  
- **DIO (Days Inventory Outstanding) / Inventory Days:** Applied to COGS to estimate inventory balance  
- **Cash Balance:** Calculated from cash inflows/outflows based on the above metrics  

All formulas and assumptions are documented in `config/settings.yaml` and the notebooks.

---

## Project Structure

```text
aum-revenue-liquidity-planning-dashboard/
├── data/
│   ├── raw/              # Untouched downloads from public sources
│   ├── external/         # Macro/industry/financial data
│   ├── interim/          # Cleaned, staged versions
│   └── processed/        # Final ML-ready tables
├── notebooks/
│   ├── 01_eda_sales.ipynb
│   ├── 02_eda_macro.ipynb
│   ├── 03_model_revenue_forecast.ipynb
│   └── 04_model_liquidity_risk.ipynb
├── src/
│   ├── data/             # Data loading and feature engineering
│   ├── models/           # ML model implementations
│   └── viz/              # Visualization utilities
├── app/
│   └── dashboard.py      # Streamlit application
├── docs/                 # Project documentation
├── config/               # Configuration files
├── reports/
│   └── figures/          # Exported charts and figures
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

Tested with: Python 3.10 on Windows 11, 8 GB RAM

### Installation

```bash
# Clone the project
git clone https://github.com/sr-dotcom/aum-revenue-liquidity-planning-dashboard.git
cd aum-revenue-liquidity-planning-dashboard

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Order

1. **(Optional)** Run `notebooks/01_eda_sales.ipynb` – EDA only
2. Run `notebooks/03_model_revenue_forecast.ipynb` – generates forecast artifacts
3. Run `notebooks/04_model_liquidity_risk.ipynb` – generates liquidity artifacts
4. Run `streamlit run app/dashboard.py` – launches dashboard

> Phases 3–4 (feature engineering and macro alignment) are covered inside the modelling notebooks and documentation.

## Running the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard opens at http://localhost:8501 with three tabs:

- **🏠 Overview** – Key metrics and combined revenue chart
- **📈 Revenue Forecast** – 12-month forecast table, scenarios, and download
- **🎯 Liquidity Risk** – Structural analysis, What-If planner, and cash simulation

**Phase 8 Planning:** After generating the Phase 5/6 artifacts, open the dashboard and experiment with What-If sliders on the Liquidity tab and scenario selectors on both Revenue and Liquidity tabs.

## Project Workflow

| Phase | Description | Notebook/File |
|---|---|---|
| 1–2 | Data acquisition & EDA | `01_eda_sales.ipynb`, `02_eda_macro.ipynb` |
| 5 | Revenue Forecasting | `03_model_revenue_forecast.ipynb` |
| 6 | Liquidity Risk Classification | `04_model_liquidity_risk.ipynb` |
| 7 | Dashboard | `app/dashboard.py` |
| 8 | Planning Enhancements | What-If, scenarios, cash simulation (UI-only) |

## Key Outputs

| Artifact | Location |
|---|---|
| 12-month forecast | `data/interim/revenue_forecast_12m.csv` |
| Trained model | `models/revenue_forecaster.pkl` |
| Liquidity risk table | `data/interim/liquidity_risk_table.csv` |

## Documentation

- **Project Charter:** `docs/project_charter.md`
- **ML Problem Definition:** `docs/ml_problem_definition.md`
- **Walkthrough:** `docs/walkthrough.md`

## License

This project is for educational and demonstration purposes only and is not intended for production use.
