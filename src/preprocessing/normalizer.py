import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_preprocessor import BasePreprocessor

class Normalizer(BasePreprocessor):
    """Normalize data using group-level scaling."""
    
    def __init__(self, method: str = 'standard', group_by: Optional[str] = None):
        super().__init__()
        self.method = method  # 'standard', 'minmax', 'robust'
        self.group_by = group_by
        self.scalers_ = {}
        self.global_scalers_ = {}
    
    def fit(self, data: pd.DataFrame, columns: List[str] = None, **kwargs) -> 'Normalizer':
        """Fit scalers."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_by:
                columns = [c for c in columns if c != self.group_by]
        
        if self.group_by and self.group_by in data.columns:
            # Group-level normalization
            for group_id in data[self.group_by].unique():
                group_data = data[data[self.group_by] == group_id]
                self.scalers_[group_id] = {}
                
                for col in columns:
                    if col in group_data.columns:
                        self._fit_scaler(group_data[col], group_id, col)
        else:
            # Global normalization
            for col in columns:
                if col in data.columns:
                    self._fit_scaler(data[col], 'global', col)
        
        self.is_fitted = True
        return self
    
    def _fit_scaler(self, series: pd.Series, group_id: str, col: str):
        """Fit scaler for a specific column and group."""
        if self.method == 'standard':
            mean = series.mean()
            std = series.std()
            if group_id == 'global':
                self.global_scalers_[col] = {'mean': mean, 'std': std}
            else:
                self.scalers_[group_id][col] = {'mean': mean, 'std': std}
        elif self.method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if group_id == 'global':
                self.global_scalers_[col] = {'min': min_val, 'max': max_val}
            else:
                self.scalers_[group_id][col] = {'min': min_val, 'max': max_val}
        elif self.method == 'robust':
            median = series.median()
            q75 = series.quantile(0.75)
            q25 = series.quantile(0.25)
            iqr = q75 - q25
            if group_id == 'global':
                self.global_scalers_[col] = {'median': median, 'iqr': iqr}
            else:
                self.scalers_[group_id][col] = {'median': median, 'iqr': iqr}
    
    def transform(self, data: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
        """Normalize data."""
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_by:
                columns = [c for c in columns if c != self.group_by]
        
        if self.group_by and self.group_by in data.columns:
            # Group-level normalization
            for group_id in data[self.group_by].unique():
                mask = data[self.group_by] == group_id
                if group_id not in self.scalers_:
                    continue
                
                for col in columns:
                    if col not in self.scalers_[group_id]:
                        continue
                    
                    scaler = self.scalers_[group_id][col]
                    data.loc[mask, col] = self._apply_scaler(data.loc[mask, col], scaler)
        else:
            # Global normalization
            for col in columns:
                if col in self.global_scalers_:
                    scaler = self.global_scalers_[col]
                    data[col] = self._apply_scaler(data[col], scaler)
        
        return data
    
    def _apply_scaler(self, series: pd.Series, scaler: Dict[str, float]) -> pd.Series:
        """Apply scaler to series."""
        if self.method == 'standard':
            return (series - scaler['mean']) / (scaler['std'] + 1e-10)
        elif self.method == 'minmax':
            return (series - scaler['min']) / (scaler['max'] - scaler['min'] + 1e-10)
        elif self.method == 'robust':
            return (series - scaler['median']) / (scaler['iqr'] + 1e-10)
        return series
    
    def report(self) -> Dict[str, Any]:
        """Generate normalization report."""
        return {
            'preprocessor': 'Normalizer',
            'method': self.method,
            'group_by': self.group_by,
            'groups_processed': len(self.scalers_),
            'is_fitted': self.is_fitted
        }

