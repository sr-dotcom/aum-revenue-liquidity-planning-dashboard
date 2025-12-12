"""
load_data.py
------------
Functions to load raw data files from various sources.

This module contains minimal, clean loader functions that:
- Read CSV files from paths specified in config
- Parse date columns as datetime
- Do NOT filter, aggregate, or drop rows (except completely empty rows)
- Return cleaned raw DataFrames for further processing

Data Policy:
- All loaders read from REAL public datasets only
- No synthetic data generation occurs in this module
- Transformations are limited to type casting and basic cleaning
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default location.
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    if config_path is None:
        # Default to config/settings.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "settings.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns
    -------
    Path
        Path to project root.
    """
    return Path(__file__).parent.parent.parent


def get_data_path(folder: str = "raw") -> Path:
    """
    Get the path to a data folder.
    
    Parameters
    ----------
    folder : str
        Subfolder name: 'raw', 'external', 'interim', or 'processed'.
    
    Returns
    -------
    Path
        Path object to the data folder.
    """
    project_root = get_project_root()
    return project_root / "data" / folder


# =============================================================================
# SALES DATA LOADER
# =============================================================================

def load_sales_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the utensil manufacturing sales dataset.
    
    Reads the CSV file at the path specified in config["data_paths"]["sales_raw"].
    Parses the date column indicated by config["columns"]["sales"]["date"] as datetime.
    Derives revenue from quantity if not present in original data.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing data_paths, columns, and finance_assumptions.
    
    Returns
    -------
    pd.DataFrame
        Sales DataFrame with:
        - Date column parsed as datetime
        - Derived revenue column (quantity * assumed_unit_price) if original lacks revenue
    
    Notes
    -----
    - This function does NOT filter or aggregate data
    - Only basic cleaning is performed (drop completely empty rows)
    - All original rows from the real dataset are preserved
    - Revenue is DERIVED using transparent formula: quantity * assumed_unit_price
    
    Example
    -------
    >>> config = load_config()
    >>> df_sales = load_sales_data(config)
    >>> print(df_sales.shape)
    """
    project_root = get_project_root()
    filepath = project_root / config["data_paths"]["sales_raw"]
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Clean column names: remove trailing/leading spaces
    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows if any
    df = df.dropna(how="all")
    
    # Parse date column
    date_col = config["columns"]["sales"]["date"]
    if date_col and not date_col.startswith("<PLACEHOLDER"):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Get quantity column name
    quantity_col = config["columns"]["sales"]["quantity"]
    revenue_col = config["columns"]["sales"]["revenue"]
    
    # Derive revenue if it's marked as derived (not in original data)
    if revenue_col == "_derived_revenue":
        # Get assumed unit price from finance assumptions
        unit_price = config["finance_assumptions"]["assumed_unit_price"]
        
        # Derive revenue = quantity * unit_price
        df["_derived_revenue"] = df[quantity_col] * unit_price
        
        print(f"[load_sales_data] Derived revenue column created: "
              f"quantity ({quantity_col}) × unit_price ({unit_price})")
    
    return df


# =============================================================================
# MACRO DATA LOADERS
# =============================================================================

def load_macro_msi_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load Manufacturers' Shipments data (FRED series AMTMVS).
    
    Reads the CSV file at config["data_paths"]["macro_msi_raw"].
    Strips column names, parses date, and returns only date + value columns.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    
    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns: date, value
    
    Notes
    -----
    - This is industry-level data from FRED
    - Provides context for manufacturing shipments
    """
    project_root = get_project_root()
    filepath = project_root / config["data_paths"]["macro_msi_raw"]
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Clean column names: remove trailing/leading spaces
    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows
    df = df.dropna(how="all")
    
    # Get column names from config
    date_col = config["columns"]["macro_msi"]["date"]
    value_col = config["columns"]["macro_msi"]["value"]
    
    # Parse date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Keep only date and value columns, rename for consistency
    df = df[[date_col, value_col]].copy()
    df.columns = ["date", "msi_value"]
    
    print(f"[load_macro_msi_data] Loaded {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def load_macro_ip_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load Industrial Production Index data (FRED series IPMAN).
    
    Reads the CSV file at config["data_paths"]["macro_ip_raw"].
    Strips column names, parses date, and returns only date + value columns.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    
    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns: date, ip_value
    
    Notes
    -----
    - This is macro-level data from FRED (series IPMAN)
    - Used as a feature for forecasting and scenario analysis
    """
    project_root = get_project_root()
    filepath = project_root / config["data_paths"]["macro_ip_raw"]
    
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Clean column names: remove trailing/leading spaces
    df.columns = df.columns.str.strip()
    
    # Drop completely empty rows
    df = df.dropna(how="all")
    
    # Get column names from config
    date_col = config["columns"]["macro_ip"]["date"]
    value_col = config["columns"]["macro_ip"]["ip_index"]
    
    # Parse date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Keep only date and value columns, rename for consistency
    df = df[[date_col, value_col]].copy()
    df.columns = ["date", "ip_value"]
    
    print(f"[load_macro_ip_data] Loaded {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


# =============================================================================
# GENERIC LOADERS
# =============================================================================

def load_csv(filepath: str, parse_dates: Optional[list] = None, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    parse_dates : list, optional
        Columns to parse as dates.
    **kwargs : dict
        Additional arguments passed to pd.read_csv().
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    df = pd.read_csv(filepath, parse_dates=parse_dates, **kwargs)
    df = df.dropna(how="all")
    return df


def load_excel(filepath: str, sheet_name: str = None, **kwargs) -> pd.DataFrame:
    """
    Load an Excel file into a pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    sheet_name : str, optional
        Name of the sheet to load.
    **kwargs : dict
        Additional arguments passed to pd.read_excel().
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    df = df.dropna(how="all")
    return df
