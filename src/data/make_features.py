"""
make_features.py
----------------
Functions to create monthly aggregated tables and engineered features.

This module contains functions for:
- Aggregating transaction data to monthly level
- Merging sales with macro data
- Creating lag and rolling features
- Building liquidity base tables with derived metrics

DATA POLICY:
- We do NOT generate synthetic historical rows
- All feature engineering is based on aggregations, joins, and formulas over real observed periods
- Months not present in original data are NOT forward-filled or fabricated
- Working capital metrics are derived using transparent formulas and documented ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# =============================================================================
# MONTHLY AGGREGATION
# =============================================================================

def make_monthly_sales(df_sales: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate transaction-level sales data to monthly level.
    
    Uses the sales date and revenue/quantity columns from config to:
    1. Extract year-month from transaction date
    2. Sum revenue and quantity for each month
    3. Return monthly aggregated DataFrame
    
    Parameters
    ----------
    df_sales : pd.DataFrame
        Raw transaction-level sales data.
    config : Dict[str, Any]
        Configuration dictionary with column mappings.
    
    Returns
    -------
    pd.DataFrame
        Monthly aggregated sales with columns:
        - year_month: Period[M]
        - total_revenue: sum of revenue for the month
        - total_quantity: sum of quantity for the month
        - transaction_count: number of transactions
    
    Notes
    -----
    - Only months present in the original data are included
    - No synthetic months are added
    - No forward/backward filling is performed
    
    Example
    -------
    >>> config = load_config()
    >>> df_sales = load_sales_data(config)
    >>> df_monthly = make_monthly_sales(df_sales, config)
    """
    df = df_sales.copy()
    
    # Get column names from config
    date_col = config["columns"]["sales"]["date"]
    revenue_col = config["columns"]["sales"]["revenue"]
    quantity_col = config["columns"]["sales"]["quantity"]
    
    # Validate columns exist (skip if placeholders)
    if date_col.startswith("<PLACEHOLDER") or revenue_col.startswith("<PLACEHOLDER"):
        raise ValueError(
            "Column placeholders must be updated in config/settings.yaml "
            "with actual column names from the dataset before running this function."
        )
    
    # Ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract year-month
    df["year_month"] = df[date_col].dt.to_period("M")
    
    # Aggregate to monthly level
    monthly_agg = df.groupby("year_month").agg(
        total_revenue=(revenue_col, "sum"),
        total_quantity=(quantity_col, "sum") if quantity_col and not quantity_col.startswith("<PLACEHOLDER") else (revenue_col, "count"),
        transaction_count=(revenue_col, "count")
    ).reset_index()
    
    # Sort by date
    monthly_agg = monthly_agg.sort_values("year_month").reset_index(drop=True)
    
    return monthly_agg


# =============================================================================
# MACRO DATA MERGE
# =============================================================================

def merge_with_macro(df_sales_monthly: pd.DataFrame,
                     df_msi: pd.DataFrame,
                     df_ip: pd.DataFrame,
                     config: Dict[str, Any]) -> pd.DataFrame:
    """
    Join monthly sales data with macro MSI and IP data on year-month.
    
    Parameters
    ----------
    df_sales_monthly : pd.DataFrame
        Monthly aggregated sales data with 'year_month' column.
    df_msi : pd.DataFrame
        Manufacturers' Sales & Inventories data.
    df_ip : pd.DataFrame
        Industrial Production Index data.
    config : Dict[str, Any]
        Configuration dictionary with column mappings.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with sales and macro data.
    
    Notes
    -----
    - Only months present in the REAL sales data are kept
    - No synthetic months are forward-filled
    - Missing macro values for existing sales months are left as NaN
    """
    df = df_sales_monthly.copy()
    
    # Get column names from config
    msi_date_col = config["columns"]["macro_msi"]["date"]
    msi_sales_col = config["columns"]["macro_msi"]["sales_index"]
    msi_inv_col = config["columns"]["macro_msi"]["inventory_index"]
    
    ip_date_col = config["columns"]["macro_ip"]["date"]
    ip_index_col = config["columns"]["macro_ip"]["ip_index"]
    
    # Process MSI data
    if not msi_date_col.startswith("<PLACEHOLDER"):
        df_msi = df_msi.copy()
        df_msi[msi_date_col] = pd.to_datetime(df_msi[msi_date_col])
        df_msi["year_month"] = df_msi[msi_date_col].dt.to_period("M")
        
        # Select relevant columns
        msi_cols = ["year_month"]
        if not msi_sales_col.startswith("<PLACEHOLDER"):
            msi_cols.append(msi_sales_col)
        if not msi_inv_col.startswith("<PLACEHOLDER"):
            msi_cols.append(msi_inv_col)
        
        df_msi_clean = df_msi[msi_cols].drop_duplicates(subset=["year_month"])
        
        # Left join to preserve only sales months
        df = df.merge(df_msi_clean, on="year_month", how="left")
    
    # Process IP data
    if not ip_date_col.startswith("<PLACEHOLDER"):
        df_ip = df_ip.copy()
        df_ip[ip_date_col] = pd.to_datetime(df_ip[ip_date_col])
        df_ip["year_month"] = df_ip[ip_date_col].dt.to_period("M")
        
        # Select relevant columns
        ip_cols = ["year_month"]
        if not ip_index_col.startswith("<PLACEHOLDER"):
            ip_cols.append(ip_index_col)
        
        df_ip_clean = df_ip[ip_cols].drop_duplicates(subset=["year_month"])
        
        # Left join to preserve only sales months
        df = df.merge(df_ip_clean, on="year_month", how="left")
    
    return df


# =============================================================================
# LAG AND ROLLING FEATURES
# =============================================================================

def create_lag_features(df: pd.DataFrame, target_col: str, 
                        lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
    """
    Create lagged features for time-series modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with time ordering.
    target_col : str
        Column to create lags for.
    lags : List[int]
        List of lag periods.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added.
    
    Notes
    -----
    - Lagged values are computed from real observed data only
    - NaN values at the beginning (where lag is not available) are preserved
    """
    df = df.copy()
    
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str,
                            windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Create rolling mean and std features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Column to create rolling features for.
    windows : List[int]
        List of rolling window sizes.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added.
    """
    df = df.copy()
    
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].rolling(window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].rolling(window).std()
    
    return df


# =============================================================================
# FORECASTING FEATURES
# =============================================================================

def build_forecasting_features(df_monthly: pd.DataFrame, 
                                df_ip: pd.DataFrame = None, 
                                df_msi: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build comprehensive features for monthly revenue forecasting.
    
    Creates time-based, lag, rolling, and macro features for ML models.
    
    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly sales data with columns:
        - year_month (Period or datetime): the month
        - total_revenue: sum of revenue for the month
        - total_quantity: sum of quantity for the month
    df_ip : pd.DataFrame, optional
        Industrial Production Index data with columns:
        - date: datetime
        - ip_value: index value
    df_msi : pd.DataFrame, optional
        Manufacturers' Shipments data with columns:
        - date: datetime
        - msi_value: shipments value
    
    Returns
    -------
    pd.DataFrame
        Feature DataFrame with:
        - month: datetime column
        - time-based features (year, month, quarter, etc.)
        - lag features (revenue_lag_1, revenue_lag_3, revenue_lag_12)
        - rolling features (revenue_roll_mean_3, revenue_roll_mean_6)
        - macro features if provided (ip_value, msi_value, changes)
        - target column 'revenue'
    
    Notes
    -----
    - Only uses past information to avoid data leakage
    - NaN values at the beginning (where lags not available) are preserved
    - Macro features are merged on year-month
    """
    df = df_monthly.copy()
    
    # Ensure we have a proper datetime column
    if 'year_month' in df.columns:
        if hasattr(df['year_month'].iloc[0], 'to_timestamp'):
            df['month'] = df['year_month'].dt.to_timestamp()
        else:
            df['month'] = pd.to_datetime(df['year_month'])
    elif 'month' not in df.columns:
        raise ValueError("DataFrame must have 'year_month' or 'month' column")
    
    # Ensure sorted by month
    df = df.sort_values('month').reset_index(drop=True)
    
    # Rename target column for clarity
    if 'total_revenue' in df.columns:
        df['revenue'] = df['total_revenue']
    
    # =================================
    # TIME-BASED FEATURES
    # =================================
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['is_year_end'] = (df['month_num'] == 12).astype(int)
    df['is_year_start'] = (df['month_num'] == 1).astype(int)
    df['is_quarter_end'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)
    
    # Sine/cosine seasonality encoding for month (captures cyclical pattern)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    
    # =================================
    # LAG FEATURES (using past data only)
    # =================================
    df['revenue_lag_1'] = df['revenue'].shift(1)
    df['revenue_lag_3'] = df['revenue'].shift(3)
    df['revenue_lag_6'] = df['revenue'].shift(6)
    df['revenue_lag_12'] = df['revenue'].shift(12)
    
    # Quantity lags
    if 'total_quantity' in df.columns:
        df['quantity_lag_1'] = df['total_quantity'].shift(1)
        df['quantity_lag_12'] = df['total_quantity'].shift(12)
    
    # =================================
    # ROLLING FEATURES (using past data only)
    # =================================
    df['revenue_roll_mean_3'] = df['revenue'].shift(1).rolling(3).mean()
    df['revenue_roll_mean_6'] = df['revenue'].shift(1).rolling(6).mean()
    df['revenue_roll_mean_12'] = df['revenue'].shift(1).rolling(12).mean()
    df['revenue_roll_std_3'] = df['revenue'].shift(1).rolling(3).std()
    df['revenue_roll_std_6'] = df['revenue'].shift(1).rolling(6).std()
    
    # Year-over-year growth features
    df['revenue_yoy_change'] = df['revenue'].shift(12)
    df['revenue_yoy_pct_change'] = (df['revenue'] - df['revenue_yoy_change']) / df['revenue_yoy_change'].replace(0, np.nan)
    
    # =================================
    # MACRO FEATURES (if provided)
    # =================================
    if df_ip is not None:
        df_ip_copy = df_ip.copy()
        df_ip_copy['month'] = pd.to_datetime(df_ip_copy['date']).dt.to_period('M').dt.to_timestamp()
        df_ip_copy = df_ip_copy[['month', 'ip_value']].drop_duplicates(subset=['month'])
        
        # Merge on month
        df = df.merge(df_ip_copy, on='month', how='left')
        
        # IP value change vs previous month
        df['ip_change'] = df['ip_value'].diff()
        df['ip_pct_change'] = df['ip_value'].pct_change()
    
    if df_msi is not None:
        df_msi_copy = df_msi.copy()
        df_msi_copy['month'] = pd.to_datetime(df_msi_copy['date']).dt.to_period('M').dt.to_timestamp()
        df_msi_copy = df_msi_copy[['month', 'msi_value']].drop_duplicates(subset=['month'])
        
        # Merge on month
        df = df.merge(df_msi_copy, on='month', how='left')
        
        # MSI value change vs previous month
        df['msi_change'] = df['msi_value'].diff()
        df['msi_pct_change'] = df['msi_value'].pct_change()
    
    # =================================
    # SELECT FINAL COLUMNS
    # =================================
    # Define feature columns (excluding identifiers and target)
    feature_cols = [
        'year', 'month_num', 'quarter', 'is_year_end', 'is_year_start', 'is_quarter_end',
        'month_sin', 'month_cos',
        'revenue_lag_1', 'revenue_lag_3', 'revenue_lag_6', 'revenue_lag_12',
        'revenue_roll_mean_3', 'revenue_roll_mean_6', 'revenue_roll_mean_12',
        'revenue_roll_std_3', 'revenue_roll_std_6',
    ]
    
    # Add quantity features if available
    if 'quantity_lag_1' in df.columns:
        feature_cols.extend(['quantity_lag_1', 'quantity_lag_12'])
    
    # Add macro features if available
    if 'ip_value' in df.columns:
        feature_cols.extend(['ip_value', 'ip_change', 'ip_pct_change'])
    if 'msi_value' in df.columns:
        feature_cols.extend(['msi_value', 'msi_change', 'msi_pct_change'])
    
    # Keep only existing columns
    available_cols = ['month'] + [c for c in feature_cols if c in df.columns] + ['revenue']
    
    return df[available_cols]


def split_time_series(df: pd.DataFrame, valid_start_date: str, 
                      date_col: str = 'month') -> tuple:
    """
    Split time series data into train and validation sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with a date column.
    valid_start_date : str
        Cutoff date string (e.g., '2019-01-01'). 
        Data before this date goes to train, on or after goes to validation.
    date_col : str
        Name of the date column (default: 'month').
    
    Returns
    -------
    tuple
        (train_df, valid_df) - DataFrames for training and validation.
    
    Example
    -------
    >>> train_df, valid_df = split_time_series(df, '2019-01-01')
    """
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    valid_start = pd.to_datetime(valid_start_date)
    
    train_df = df[df[date_col] < valid_start].copy()
    valid_df = df[df[date_col] >= valid_start].copy()
    
    return train_df, valid_df


# =============================================================================
# LIQUIDITY BASE TABLE
# =============================================================================

def build_liquidity_base_table(df_monthly: pd.DataFrame, 
                                config: Dict[str, Any],
                                df_forecast: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a base liquidity table combining historical monthly sales
    and optional forecast data, then compute working-capital and liquidity metrics.
    
    Parameters
    ----------
    df_monthly : pd.DataFrame
        Historical monthly sales data with columns:
        - year_month or date: the month identifier
        - total_revenue: monthly revenue
        - total_quantity: monthly quantity (optional)
    config : Dict[str, Any]
        Configuration dictionary with finance_assumptions and liquidity_thresholds.
    df_forecast : pd.DataFrame, optional
        Forecast data with columns:
        - month or year_month: the month identifier
        - revenue_forecast: forecasted revenue
    
    Returns
    -------
    pd.DataFrame
        Liquidity base table with one row per month and columns:
        - year_month: the month
        - revenue: actual or forecast revenue
        - is_forecast: boolean flag
        - cogs: cost of goods sold
        - gross_margin_pct: gross margin percentage
        - dso_days, dpo_days, inventory_days: working capital assumptions
        - ccc_days: cash conversion cycle
        - operating_cash_margin: gross margin - fixed costs
        - operating_cash_flow: revenue * operating_cash_margin
        - adjusted_liquidity_score: final score for risk classification
    
    Notes
    -----
    - Historical months use actual revenue
    - Forecast months use revenue_forecast
    - All assumptions are from config/settings.yaml
    - Result is saved to data/interim/liquidity_base_table.csv
    """
    from pathlib import Path
    
    # Get finance assumptions from config
    gross_margin_pct = config["finance_assumptions"]["gross_margin_pct"]
    dso_days = config["finance_assumptions"]["dso_days"]
    dpo_days = config["finance_assumptions"]["dpo_days"]
    inventory_days = config["finance_assumptions"]["inventory_days"]
    fixed_cost_ratio = config["finance_assumptions"].get("fixed_cost_ratio", 0.15)
    wc_penalty = config["finance_assumptions"].get("wc_penalty", 0.2)
    
    # Prepare historical data
    df_hist = df_monthly.copy()
    
    # Normalize column names
    if 'year_month' in df_hist.columns:
        df_hist['year_month'] = pd.to_datetime(df_hist['year_month'])
    elif 'date' in df_hist.columns:
        df_hist['year_month'] = pd.to_datetime(df_hist['date'])
    
    # Use total_revenue or revenue column
    if 'total_revenue' in df_hist.columns:
        df_hist['revenue'] = df_hist['total_revenue']
    elif 'revenue' not in df_hist.columns:
        raise ValueError("DataFrame must have 'total_revenue' or 'revenue' column")
    
    df_hist['is_forecast'] = False
    df_hist = df_hist[['year_month', 'revenue', 'is_forecast']].copy()
    
    # Prepare forecast data if provided
    if df_forecast is not None:
        df_fc = df_forecast.copy()
        
        # Normalize column names
        if 'month' in df_fc.columns:
            df_fc['year_month'] = pd.to_datetime(df_fc['month'])
        elif 'year_month' in df_fc.columns:
            df_fc['year_month'] = pd.to_datetime(df_fc['year_month'])
        
        if 'revenue_forecast' in df_fc.columns:
            df_fc['revenue'] = df_fc['revenue_forecast']
        
        df_fc['is_forecast'] = True
        df_fc = df_fc[['year_month', 'revenue', 'is_forecast']].copy()
        
        # Combine: historical + forecast (avoiding duplicates)
        max_hist_date = df_hist['year_month'].max()
        df_fc = df_fc[df_fc['year_month'] > max_hist_date]
        
        df = pd.concat([df_hist, df_fc], ignore_index=True)
    else:
        df = df_hist
    
    # Sort by date
    df = df.sort_values('year_month').reset_index(drop=True)
    
    # =================================
    # COMPUTE LIQUIDITY METRICS
    # =================================
    
    # COGS = revenue * (1 - gross_margin_pct)
    df['cogs'] = df['revenue'] * (1 - gross_margin_pct)
    
    # Store assumptions as columns for transparency
    df['gross_margin_pct'] = gross_margin_pct
    df['dso_days'] = dso_days
    df['dpo_days'] = dpo_days
    df['inventory_days'] = inventory_days
    
    # Cash Conversion Cycle = DSO + Inventory Days - DPO
    df['ccc_days'] = dso_days + inventory_days - dpo_days
    
    # Operating Cash Margin = Gross Margin - Fixed Cost Ratio
    df['operating_cash_margin'] = gross_margin_pct - fixed_cost_ratio
    
    # Operating Cash Flow = Revenue * Operating Cash Margin
    df['operating_cash_flow'] = df['revenue'] * df['operating_cash_margin']
    
    # Adjusted Liquidity Score
    # Score = operating_cash_margin - (ccc_days / 365) * wc_penalty
    # Higher score = better liquidity, lower risk
    df['adjusted_liquidity_score'] = (
        df['operating_cash_margin'] - (df['ccc_days'] / 365.0) * wc_penalty
    )
    
    # Select final columns
    output_cols = [
        'year_month', 'revenue', 'is_forecast', 'cogs',
        'gross_margin_pct', 'dso_days', 'dpo_days', 'inventory_days', 'ccc_days',
        'operating_cash_margin', 'operating_cash_flow', 'adjusted_liquidity_score'
    ]
    df = df[output_cols]
    
    # Save to CSV
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / 'data' / 'interim' / 'liquidity_base_table.csv'
    df.to_csv(output_path, index=False)
    print(f"[build_liquidity_base_table] Saved {len(df)} rows to {output_path}")
    
    return df


# =============================================================================
# WORKING CAPITAL METRIC CALCULATIONS
# =============================================================================

def calculate_dso(revenue: float, accounts_receivable: float, days: int = 30) -> float:
    """
    Calculate Days Sales Outstanding (DSO).
    
    DSO = (Accounts Receivable / Revenue) * Days
    
    Parameters
    ----------
    revenue : float
        Revenue for the period.
    accounts_receivable : float
        Accounts receivable balance.
    days : int
        Number of days in the period (default 30 for monthly).
    
    Returns
    -------
    float
        DSO value.
    """
    if revenue == 0:
        return np.nan
    return (accounts_receivable / revenue) * days


def calculate_dpo(cogs: float, accounts_payable: float, days: int = 30) -> float:
    """
    Calculate Days Payable Outstanding (DPO).
    
    DPO = (Accounts Payable / COGS) * Days
    
    Parameters
    ----------
    cogs : float
        Cost of goods sold for the period.
    accounts_payable : float
        Accounts payable balance.
    days : int
        Number of days in the period (default 30 for monthly).
    
    Returns
    -------
    float
        DPO value.
    """
    if cogs == 0:
        return np.nan
    return (accounts_payable / cogs) * days


def calculate_cash_conversion_cycle(dso: float, dio: float, dpo: float) -> float:
    """
    Calculate Cash Conversion Cycle (CCC).
    
    CCC = DSO + DIO - DPO
    
    Parameters
    ----------
    dso : float
        Days Sales Outstanding.
    dio : float
        Days Inventory Outstanding.
    dpo : float
        Days Payable Outstanding.
    
    Returns
    -------
    float
        Cash Conversion Cycle value.
    """
    return dso + dio - dpo
