"""
liquidity.py
------------
Liquidity risk classification models.

This module contains:
- LiquidityRiskClassifier: Rule-based classifier for Safe / At Risk / Critical
- Helper functions for liquidity classification
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from enum import Enum


class LiquidityRiskLevel(Enum):
    """Liquidity risk classification levels."""
    SAFE = "Safe"
    AT_RISK = "At Risk"
    CRITICAL = "Critical"


# =============================================================================
# RULE-BASED LIQUIDITY RISK CLASSIFIER
# =============================================================================

class LiquidityRiskClassifier:
    """
    Rule-based liquidity risk classifier.
    
    Classifies each month into Safe / At Risk / Critical based on the
    adjusted_liquidity_score computed in build_liquidity_base_table().
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with liquidity_thresholds:
        - safe_min_margin: score >= this → Safe
        - at_risk_min_margin: score in [this, safe_min_margin) → At Risk
        - score < at_risk_min_margin → Critical
    
    Example
    -------
    >>> from data.load_data import load_config
    >>> config = load_config()
    >>> clf = LiquidityRiskClassifier(config)
    >>> df_scored = clf.score(df_liquidity)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize classifier with thresholds from config."""
        thresholds = config.get("liquidity_thresholds", {})
        self.safe_min = thresholds.get("safe_min_margin", 0.15)
        self.at_risk_min = thresholds.get("at_risk_min_margin", 0.05)
    
    def score_row(self, row: pd.Series) -> str:
        """
        Classify a single row based on adjusted_liquidity_score.
        
        Parameters
        ----------
        row : pd.Series
            Row containing 'adjusted_liquidity_score' column.
        
        Returns
        -------
        str
            Risk label: "Safe", "At Risk", or "Critical"
        """
        s = row["adjusted_liquidity_score"]
        
        if s >= self.safe_min:
            return "Safe"
        elif s >= self.at_risk_min:
            return "At Risk"
        else:
            return "Critical"
    
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all rows in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Liquidity base table with 'adjusted_liquidity_score' column.
        
        Returns
        -------
        pd.DataFrame
            Input DataFrame with added 'liquidity_risk_label' column.
        """
        df = df.copy()
        df["liquidity_risk_label"] = df.apply(self.score_row, axis=1)
        return df
    
    def get_thresholds(self) -> Dict[str, float]:
        """Return the thresholds used for classification."""
        return {
            "safe_min_margin": self.safe_min,
            "at_risk_min_margin": self.at_risk_min
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_liquidity(df_liquidity: pd.DataFrame, 
                       config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply liquidity risk classification to a liquidity base table.
    
    This is a convenience wrapper around LiquidityRiskClassifier.
    
    Parameters
    ----------
    df_liquidity : pd.DataFrame
        Liquidity base table from build_liquidity_base_table().
        Must contain 'adjusted_liquidity_score' column.
    config : Dict[str, Any]
        Configuration dictionary with liquidity_thresholds.
    
    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'liquidity_risk_label' column.
    
    Example
    -------
    >>> df_scored = classify_liquidity(df_liquidity, config)
    >>> print(df_scored['liquidity_risk_label'].value_counts())
    """
    clf = LiquidityRiskClassifier(config)
    return clf.score(df_liquidity)


def get_risk_summary(df_scored: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of liquidity risk classification results.
    
    Parameters
    ----------
    df_scored : pd.DataFrame
        Scored DataFrame with 'liquidity_risk_label' column.
    
    Returns
    -------
    Dict[str, Any]
        Summary statistics including counts, percentages, and trends.
    """
    if 'liquidity_risk_label' not in df_scored.columns:
        raise ValueError("DataFrame must have 'liquidity_risk_label' column")
    
    counts = df_scored['liquidity_risk_label'].value_counts().to_dict()
    total = len(df_scored)
    
    # Calculate percentages
    pcts = {k: v / total * 100 for k, v in counts.items()}
    
    # Separate historical vs forecast
    if 'is_forecast' in df_scored.columns:
        hist_counts = df_scored[~df_scored['is_forecast']]['liquidity_risk_label'].value_counts().to_dict()
        fc_counts = df_scored[df_scored['is_forecast']]['liquidity_risk_label'].value_counts().to_dict()
    else:
        hist_counts = counts
        fc_counts = {}
    
    return {
        'total_months': total,
        'label_counts': counts,
        'label_percentages': pcts,
        'historical_counts': hist_counts,
        'forecast_counts': fc_counts
    }
