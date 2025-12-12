# Data Inventory: Aurora Utensils Manufacturing

This document catalogs all datasets used in the AUM Forecasting & Liquidity Planning project.

> **Data Policy**: All base data must originate from real public sources. No synthetic transactional rows are permitted. We may clean, aggregate, and join tables, but we do NOT fabricate new orders or customers.

---

## Dataset Inventory

| Dataset Name | Source / Provider | Time Coverage | Granularity | Role in Project | Status |
|-------------|-------------------|---------------|-------------|-----------------|--------|
| **Sales Dataset of a Medium-Scale Utensil Manufacturing Company** | Kaggle | 2016-01 to 2020-12 | Transaction-level | Core transaction history; saved at `data/raw/utensil_sales.csv` | ✅ Downloaded |
| **Manufacturers' Shipments – Total Manufacturing** | FRED (AMTMVS) | Monthly, multi-year | Industry-level | Manufacturing shipments for macro context; save to `data/external/mfg_shipments.csv` | 🔲 To Download |
| **Industrial Production Index – Manufacturing** | FRED (IPMAN) | Monthly, multi-year | Macro index | Manufacturing cycle feature; save to `data/external/ipman.csv` | 🔲 To Download |
| **SEC Financial Statement Data Sets** | SEC EDGAR | Annual/Quarterly | Firm-level | Realistic DSO/DPO/margin ranges for liquidity simulation | 🔲 Optional |

---

## Dataset Details

### 1. Sales Dataset of a Medium-Scale Utensil Manufacturing Company (Core)

- **Purpose**: Serves as Aurora Utensils Manufacturing's transaction history
- **Required Columns**: Date, Product/SKU, Region or Customer, Quantity, Revenue/Sales Amount
- **Expected Content**: Multi-year transactional sales data from a utensil manufacturing company
- **Notes**: 
  - This is a real manufacturing sales dataset, NOT a retail/e-commerce dataset
  - We will aggregate to monthly granularity for time-series forecasting
  - Product categories should reflect utensils/kitchenware manufacturing

### 2. Manufacturers' Shipments – Total Manufacturing (AMTMVS)

- **FRED Series ID**: AMTMVS
- **Source URL**: https://fred.stlouisfed.org/series/AMTMVS
- **Download Location**: `data/external/mfg_shipments.csv`
- **Purpose**: Total value of manufacturers' shipments for the manufacturing sector
- **Expected Columns**: DATE, AMTMVS (or VALUE)
- **Notes**: 
  - Seasonally adjusted monthly data
  - Use 2016-01 to 2020-12 window to match sales data
  - Provides context for Aurora's performance relative to industry

### 3. Industrial Production Index – Manufacturing (IPMAN)

- **FRED Series ID**: IPMAN
- **Source URL**: https://fred.stlouisfed.org/series/IPMAN
- **Download Location**: `data/external/ipman.csv`
- **Purpose**: Captures manufacturing sector business cycles
- **Expected Columns**: DATE, IPMAN (or VALUE)
- **Notes**: 
  - Consider both index levels and YoY growth rates as features
  - Use 2016-01 to 2020-12 window to match sales data
  - Useful for macro-adjusted forecasting and scenario analysis

### 4. SEC Financial Statement Data Sets (Optional)

- **Source URL**: https://www.sec.gov/dera/data/financial-statement-data-sets
- **Purpose**: Validate and calibrate DSO, DPO, gross margin, inventory turnover assumptions
- **Candidates**:
  - SEC 10-K filings from comparable manufacturing companies
  - XBRL extracts with standardized financial metrics
- **Notes**: Used to derive realistic ranges for working capital simulation, not as direct model input

---

## Data Pipeline Status

| Stage | Description | Status |
|-------|-------------|--------|
| **Raw** (`data/raw/`) | Original downloaded files | ✅ Sales data loaded |
| **External** (`data/external/`) | Macro/industry data files | 🔲 Pending FRED downloads |
| **Interim** (`data/interim/`) | Cleaned and staged data | ✅ monthly_sales.csv created |
| **Processed** (`data/processed/`) | ML-ready feature tables | 🔲 Pending |

---

## Next Steps

1. ✅ ~~Download the utensil manufacturing sales dataset from Kaggle into `data/raw/`~~
2. 🔲 Download IPMAN from FRED → save as `data/external/ipman.csv`
3. 🔲 Download AMTMVS from FRED → save as `data/external/mfg_shipments.csv`
4. 🔲 Complete EDA in `notebooks/02_eda_macro.ipynb`
5. 🔲 Merge macro data with monthly sales for forecasting features
