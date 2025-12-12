"""
plots.py
--------
Reusable plotting utilities for EDA and dashboard visualizations.

This module will contain:
- Time-series plots (actual vs. forecast)
- Liquidity gauge/indicator visualizations
- Distribution and trend charts
- Macro overlay plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


# ============================================================================
# Plot styling defaults
# ============================================================================

def set_plot_style():
    """Set consistent plot styling for the project."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


# ============================================================================
# Time-series visualizations
# ============================================================================

def plot_time_series(data: pd.DataFrame, date_col: str, value_col: str,
                     title: str = "Time Series", 
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a basic time-series.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    date_col : str
        Name of the date column.
    value_col : str
        Name of the value column.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data[date_col], data[value_col], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_forecast_vs_actual(actual: pd.Series, forecast: pd.Series,
                            dates: pd.Series,
                            title: str = "Forecast vs Actual",
                            confidence_interval: Optional[Tuple] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot forecast against actual values.
    
    Parameters
    ----------
    actual : pd.Series
        Actual values.
    forecast : pd.Series
        Forecasted values.
    dates : pd.Series
        Date index.
    title : str
        Plot title.
    confidence_interval : Tuple, optional
        Lower and upper bounds for confidence interval.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, actual, label='Actual', linewidth=2, color='blue')
    ax.plot(dates, forecast, label='Forecast', linewidth=2, 
            color='red', linestyle='--')
    
    if confidence_interval:
        ax.fill_between(dates, confidence_interval[0], confidence_interval[1],
                        alpha=0.2, color='red', label='95% CI')
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ============================================================================
# Liquidity visualizations
# ============================================================================

def plot_liquidity_gauge(value: float, metric_name: str = "Current Ratio",
                          thresholds: Tuple[float, float] = (1.0, 1.5),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a gauge-style visualization for liquidity metrics.
    (Placeholder - will be implemented with Plotly for interactivity)
    
    Parameters
    ----------
    value : float
        Current metric value.
    metric_name : str
        Name of the metric.
    thresholds : Tuple[float, float]
        (at_risk_threshold, safe_threshold).
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Placeholder implementation
    raise NotImplementedError("plot_liquidity_gauge() not yet implemented")


def plot_cash_flow_waterfall(inflows: List[float], outflows: List[float],
                              labels: List[str],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a waterfall chart for cash flow visualization.
    (Placeholder - to be implemented)
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Placeholder implementation
    raise NotImplementedError("plot_cash_flow_waterfall() not yet implemented")


# ============================================================================
# Distribution and trend charts
# ============================================================================

def plot_distribution(data: pd.Series, title: str = "Distribution",
                      bins: int = 30,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram with KDE.
    
    Parameters
    ----------
    data : pd.Series
        Data to plot.
    title : str
        Plot title.
    bins : int
        Number of histogram bins.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(data: pd.DataFrame, 
                              title: str = "Correlation Matrix",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a correlation heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to compute correlations for.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
