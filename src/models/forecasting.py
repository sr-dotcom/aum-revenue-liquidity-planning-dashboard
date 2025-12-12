"""
forecasting.py
--------------
Production-ready revenue forecasting class for Aurora Utensils Manufacturing.

This module provides:
- AuroraRevenueForecaster: Main forecasting class with fit/predict interface
- Helper functions for loading/saving trained models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import joblib

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# AURORA REVENUE FORECASTER CLASS
# =============================================================================

class AuroraRevenueForecaster:
    """
    Revenue forecasting model for Aurora Utensils Manufacturing.
    
    This class provides a complete pipeline for:
    - Building forecasting features from monthly sales data
    - Training ML models (Gradient Boosting or Random Forest)
    - Generating multi-step forecasts
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with model settings.
    model_type : str, optional
        Type of model to use: 'gradient_boosting' or 'random_forest'.
        Default: 'gradient_boosting'
    horizon : int, optional
        Default forecast horizon in months. Default: 12
    
    Attributes
    ----------
    model : sklearn estimator
        Fitted model object.
    feature_cols : List[str]
        List of feature column names used by the model.
    is_fitted : bool
        Whether the model has been fitted.
    
    Example
    -------
    >>> from data.load_data import load_config
    >>> config = load_config()
    >>> forecaster = AuroraRevenueForecaster(config)
    >>> forecaster.fit(df_monthly, df_ip, df_msi)
    >>> forecast = forecaster.predict(12)
    """
    
    def __init__(self, config: Dict[str, Any], 
                 model_type: str = 'gradient_boosting',
                 horizon: int = 12):
        self.config = config
        self.model_type = model_type
        self.horizon = horizon
        self.model = None
        self.feature_cols = None
        self.is_fitted = False
        self._history = None
        self._df_ip = None
        self._df_msi = None
        self._validation_metrics = None
        
    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _build_features(self, df_monthly: pd.DataFrame, 
                        df_ip: pd.DataFrame = None,
                        df_msi: pd.DataFrame = None) -> pd.DataFrame:
        """Build forecasting features from monthly data."""
        # Import here to avoid circular imports
        from data.make_features import build_forecasting_features
        return build_forecasting_features(df_monthly, df_ip, df_msi)
    
    def fit(self, df_monthly: pd.DataFrame, 
            df_ip: pd.DataFrame = None, 
            df_msi: pd.DataFrame = None,
            valid_start_date: str = '2019-01-01') -> 'AuroraRevenueForecaster':
        """
        Fit the forecaster on monthly sales data.
        
        Parameters
        ----------
        df_monthly : pd.DataFrame
            Monthly sales data with columns [year_month, total_revenue, total_quantity].
        df_ip : pd.DataFrame, optional
            Industrial Production Index data.
        df_msi : pd.DataFrame, optional
            Manufacturers' Shipments data.
        valid_start_date : str, optional
            Date to split train/validation. Default: '2019-01-01'
        
        Returns
        -------
        self
            Fitted forecaster instance.
        """
        # Store references for prediction
        self._df_ip = df_ip
        self._df_msi = df_msi
        
        # Build features
        df_features = self._build_features(df_monthly, df_ip, df_msi)
        
        # Drop NaN rows (from lag features)
        df_clean = df_features.dropna().copy()
        
        # Store history for future predictions
        self._history = df_clean.copy()
        
        # Split train/validation
        from data.make_features import split_time_series
        train_df, valid_df = split_time_series(df_clean, valid_start_date)
        
        # Define features and target
        self.feature_cols = [c for c in df_clean.columns if c not in ['month', 'revenue']]
        target_col = 'revenue'
        
        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col]
        X_valid = valid_df[self.feature_cols]
        y_valid = valid_df[target_col]
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Calculate validation metrics
        y_pred = self.model.predict(X_valid)
        self._validation_metrics = {
            'MAE': mean_absolute_error(y_valid, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_valid, y_pred)),
            'MAPE': np.mean(np.abs((y_valid - y_pred) / y_valid.replace(0, np.nan))) * 100
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, n_months: int = None,
                df_ip_future: pd.DataFrame = None,
                df_msi_future: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate n-month ahead revenue forecast.
        
        Parameters
        ----------
        n_months : int, optional
            Number of months to forecast. Default: self.horizon (12)
        df_ip_future : pd.DataFrame, optional
            Future Industrial Production data (if available).
        df_msi_future : pd.DataFrame, optional
            Future Manufacturers' Shipments data (if available).
        
        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with columns [month, revenue_forecast].
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if n_months is None:
            n_months = self.horizon
        
        # Use provided future macro data or fall back to stored historical
        df_ip = df_ip_future if df_ip_future is not None else self._df_ip
        df_msi = df_msi_future if df_msi_future is not None else self._df_msi
        
        # Get last date from history
        last_date = self._history['month'].max()
        
        # Generate predictions
        future_predictions = []
        all_revenue = list(self._history['revenue'].values)
        
        for i in range(n_months):
            next_month = last_date + pd.DateOffset(months=i+1)
            
            # Create feature row
            row = self._create_future_features(next_month, all_revenue, df_ip, df_msi)
            
            # Make prediction
            X_future = pd.DataFrame([row])[self.feature_cols]
            pred = self.model.predict(X_future)[0]
            
            # Store prediction
            future_predictions.append({
                'month': next_month,
                'revenue_forecast': pred
            })
            
            # Add to revenue history for next iteration
            all_revenue.append(pred)
        
        return pd.DataFrame(future_predictions)
    
    def _create_future_features(self, next_month, all_revenue: List[float],
                                df_ip: pd.DataFrame, df_msi: pd.DataFrame) -> Dict:
        """Create feature dictionary for a future month."""
        row = {
            'year': next_month.year,
            'month_num': next_month.month,
            'quarter': (next_month.month - 1) // 3 + 1,
            'is_year_end': 1 if next_month.month == 12 else 0,
            'is_year_start': 1 if next_month.month == 1 else 0,
            'is_quarter_end': 1 if next_month.month in [3, 6, 9, 12] else 0,
            'month_sin': np.sin(2 * np.pi * next_month.month / 12),
            'month_cos': np.cos(2 * np.pi * next_month.month / 12),
        }
        
        # Lag features
        row['revenue_lag_1'] = all_revenue[-1] if len(all_revenue) >= 1 else np.nan
        row['revenue_lag_3'] = all_revenue[-3] if len(all_revenue) >= 3 else np.nan
        row['revenue_lag_6'] = all_revenue[-6] if len(all_revenue) >= 6 else np.nan
        row['revenue_lag_12'] = all_revenue[-12] if len(all_revenue) >= 12 else np.nan
        
        # Rolling features
        if len(all_revenue) >= 4:
            row['revenue_roll_mean_3'] = np.mean(all_revenue[-4:-1])
            row['revenue_roll_std_3'] = np.std(all_revenue[-4:-1])
        else:
            row['revenue_roll_mean_3'] = np.nan
            row['revenue_roll_std_3'] = np.nan
            
        if len(all_revenue) >= 7:
            row['revenue_roll_mean_6'] = np.mean(all_revenue[-7:-1])
            row['revenue_roll_std_6'] = np.std(all_revenue[-7:-1])
        else:
            row['revenue_roll_mean_6'] = np.nan
            row['revenue_roll_std_6'] = np.nan
            
        if len(all_revenue) >= 13:
            row['revenue_roll_mean_12'] = np.mean(all_revenue[-13:-1])
        else:
            row['revenue_roll_mean_12'] = np.nan
        
        # Quantity features (use average from history)
        if 'quantity_lag_1' in self.feature_cols:
            avg_qty = self._history['total_quantity'].mean() if 'total_quantity' in self._history.columns else 0
            row['quantity_lag_1'] = avg_qty
            row['quantity_lag_12'] = avg_qty
        
        # Macro features
        if df_ip is not None and 'ip_value' in self.feature_cols:
            mask = df_ip['date'] <= next_month
            last_ip = df_ip[mask].iloc[-1] if mask.any() else df_ip.iloc[-1]
            row['ip_value'] = last_ip['ip_value']
            row['ip_change'] = 0
            row['ip_pct_change'] = 0
            
        if df_msi is not None and 'msi_value' in self.feature_cols:
            mask = df_msi['date'] <= next_month
            last_msi = df_msi[mask].iloc[-1] if mask.any() else df_msi.iloc[-1]
            row['msi_value'] = last_msi['msi_value']
            row['msi_change'] = 0
            row['msi_pct_change'] = 0
        
        return row
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """Get validation metrics from training."""
        if self._validation_metrics is None:
            raise RuntimeError("No validation metrics available. Call fit() first.")
        return self._validation_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: str):
        """Save the trained forecaster to a file."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Cannot save.")
        
        save_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'horizon': self.horizon,
            'validation_metrics': self._validation_metrics,
            'history': self._history,
            'config': self.config,
            'df_ip': self._df_ip,
            'df_msi': self._df_msi
        }
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'AuroraRevenueForecaster':
        """Load a trained forecaster from a file."""
        save_data = joblib.load(filepath)
        
        forecaster = cls(
            config=save_data['config'],
            model_type=save_data['model_type'],
            horizon=save_data['horizon']
        )
        forecaster.model = save_data['model']
        forecaster.feature_cols = save_data['feature_cols']
        forecaster._validation_metrics = save_data['validation_metrics']
        forecaster._history = save_data['history']
        forecaster._df_ip = save_data.get('df_ip')
        forecaster._df_msi = save_data.get('df_msi')
        forecaster.is_fitted = True
        
        return forecaster


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_trained_forecaster(config: Dict[str, Any]) -> AuroraRevenueForecaster:
    """
    Load a trained forecaster from the default model path.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary (used to find model path).
    
    Returns
    -------
    AuroraRevenueForecaster
        Loaded forecaster instance.
    """
    # Find project root
    from data.load_data import get_project_root
    project_root = get_project_root()
    model_path = project_root / 'models' / 'revenue_forecaster.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return AuroraRevenueForecaster.load(str(model_path))


def calculate_forecast_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecasting metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
