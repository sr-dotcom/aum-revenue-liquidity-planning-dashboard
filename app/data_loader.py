"""
data_loader.py
---------------
Data loading utilities for the AUM dashboard.

Provides helper functions to load all Phase 5/6 artifacts for dashboard display.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is in app/, so parent is project root
    return Path(__file__).parent.parent


def load_config() -> Dict[str, Any]:
    """Load project configuration from settings.yaml."""
    config_path = get_project_root() / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_revenue_history_and_forecast() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical monthly revenue and 12-month forecast.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_history, df_forecast)
        - df_history: Historical monthly sales with 'year_month', 'revenue' columns
        - df_forecast: 12-month forecast with 'year_month', 'revenue' columns
    """
    root = get_project_root()
    
    # Load historical data
    hist_path = root / "data" / "interim" / "monthly_sales.csv"
    if not hist_path.exists():
        raise FileNotFoundError(
            f"Historical data not found: {hist_path}\n"
            "Run Phase 5 notebook to generate this file."
        )
    
    df_hist = pd.read_csv(hist_path)
    df_hist['year_month'] = pd.to_datetime(df_hist['year_month'])
    
    # Normalize column name
    if 'total_revenue' in df_hist.columns:
        df_hist = df_hist.rename(columns={'total_revenue': 'revenue'})
    
    # Load forecast data
    fc_path = root / "data" / "interim" / "revenue_forecast_12m.csv"
    if not fc_path.exists():
        raise FileNotFoundError(
            f"Forecast data not found: {fc_path}\n"
            "Run Phase 5 notebook to generate this file."
        )
    
    df_fc = pd.read_csv(fc_path)
    
    # Normalize column names
    if 'month' in df_fc.columns:
        df_fc = df_fc.rename(columns={'month': 'year_month'})
    if 'revenue_forecast' in df_fc.columns:
        df_fc = df_fc.rename(columns={'revenue_forecast': 'revenue'})
    
    df_fc['year_month'] = pd.to_datetime(df_fc['year_month'])
    
    return df_hist, df_fc


def get_liquidity_base_table() -> pd.DataFrame:
    """
    Load the liquidity base table (historical + forecast with liquidity metrics).
    
    Returns
    -------
    pd.DataFrame
        Liquidity base table with columns:
        year_month, revenue, is_forecast, cogs, gross_margin_pct, 
        dso_days, dpo_days, inventory_days, ccc_days,
        operating_cash_margin, operating_cash_flow, adjusted_liquidity_score
    """
    root = get_project_root()
    path = root / "data" / "interim" / "liquidity_base_table.csv"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Liquidity base table not found: {path}\n"
            "Run Phase 6 notebook to generate this file."
        )
    
    df = pd.read_csv(path)
    df['year_month'] = pd.to_datetime(df['year_month'])
    return df


def get_liquidity_risk_table() -> pd.DataFrame:
    """
    Load the liquidity risk table with risk classifications.
    
    Returns
    -------
    pd.DataFrame
        Risk table with all liquidity base columns plus 'liquidity_risk_label'
    """
    root = get_project_root()
    path = root / "data" / "interim" / "liquidity_risk_table.csv"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Liquidity risk table not found: {path}\n"
            "Run Phase 6 notebook to generate this file."
        )
    
    df = pd.read_csv(path)
    df['year_month'] = pd.to_datetime(df['year_month'])
    return df


def get_liquidity_config_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of liquidity configuration and current score.
    
    Returns all structural parameters needed for what-if planning calculations.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - gross_margin_pct, fixed_cost_ratio, wc_penalty (from config)
        - dso_days, dio_days, dpo_days (from config)
        - ccc_days: Cash Conversion Cycle (computed)
        - operating_cash_margin: Operating cash margin (computed)
        - score: Adjusted liquidity score (from data)
        - safe_threshold, at_risk_threshold (from config)
        - risk_band: Risk classification label
    """
    config = load_config()
    df_risk = get_liquidity_risk_table()
    
    # Get first row values (structural values are constant across all rows)
    first_row = df_risk.iloc[0]
    
    # Get finance assumptions from config
    fin = config.get('finance_assumptions', {})
    gross_margin_pct = fin.get('gross_margin_pct', 0.30)
    fixed_cost_ratio = fin.get('fixed_cost_ratio', 0.15)
    wc_penalty = fin.get('wc_penalty', 0.20)
    dso_days = fin.get('dso_days', 45)
    dio_days = fin.get('inventory_days', 60)
    dpo_days = fin.get('dpo_days', 40)
    
    # Get thresholds from config
    thresholds = config.get('liquidity_thresholds', {})
    safe_min = thresholds.get('safe_min_margin', 0.15)
    at_risk_min = thresholds.get('at_risk_min_margin', 0.05)
    
    return {
        # Config values (for what-if sliders)
        'gross_margin_pct': gross_margin_pct,
        'fixed_cost_ratio': fixed_cost_ratio,
        'wc_penalty': wc_penalty,
        'dso_days': dso_days,
        'dio_days': dio_days,
        'dpo_days': dpo_days,
        
        # Computed structural values
        'ccc_days': first_row['ccc_days'],
        'operating_cash_margin': first_row['operating_cash_margin'],
        'score': first_row['adjusted_liquidity_score'],
        'risk_band': first_row['liquidity_risk_label'],
        
        # Thresholds
        'safe_threshold': safe_min,
        'at_risk_threshold': at_risk_min,
    }


def get_forecast_csv_bytes() -> bytes:
    """Get the forecast CSV as bytes for download button."""
    root = get_project_root()
    path = root / "data" / "interim" / "revenue_forecast_12m.csv"
    
    if not path.exists():
        return b""
    
    with open(path, 'rb') as f:
        return f.read()


def get_liquidity_csv_bytes() -> bytes:
    """Get the liquidity risk table CSV as bytes for download button."""
    root = get_project_root()
    path = root / "data" / "interim" / "liquidity_risk_table.csv"
    
    if not path.exists():
        return b""
    
    with open(path, 'rb') as f:
        return f.read()


def check_data_availability() -> Dict[str, bool]:
    """
    Check which data files are available.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary with file availability status
    """
    root = get_project_root()
    
    return {
        'monthly_sales': (root / "data" / "interim" / "monthly_sales.csv").exists(),
        'revenue_forecast': (root / "data" / "interim" / "revenue_forecast_12m.csv").exists(),
        'liquidity_base': (root / "data" / "interim" / "liquidity_base_table.csv").exists(),
        'liquidity_risk': (root / "data" / "interim" / "liquidity_risk_table.csv").exists(),
    }
