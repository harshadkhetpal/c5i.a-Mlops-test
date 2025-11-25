import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base_preprocessor import BasePreprocessor

class OutlierHandler(BasePreprocessor):
    """Detect and handle outliers using IQR and Z-score methods."""
    
    def __init__(self, method: str = 'iqr', threshold: float = 3.0, 
                 action: str = 'clip'):
        super().__init__()
        self.method = method  # 'iqr', 'zscore', 'both'
        self.threshold = threshold
        self.action = action  # 'clip', 'remove', 'mark'
        self.outlier_stats_ = {}
        self.bounds_ = {}
    
    def fit(self, data: pd.DataFrame, columns: List[str] = None, **kwargs) -> 'OutlierHandler':
        """Calculate outlier bounds."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.bounds_ = {}
        self.outlier_stats_ = {}
        
        for col in columns:
            if self.method in ['iqr', 'both']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.bounds_[f'{col}_iqr'] = (lower_bound, upper_bound)
            
            if self.method in ['zscore', 'both']:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    lower_bound = mean - self.threshold * std
                    upper_bound = mean + self.threshold * std
                    self.bounds_[f'{col}_zscore'] = (lower_bound, upper_bound)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, columns: List[str] = None, **kwargs) -> pd.DataFrame:
        """Handle outliers."""
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_counts = {}
        
        for col in columns:
            if self.method == 'iqr' and f'{col}_iqr' in self.bounds_:
                lower, upper = self.bounds_[f'{col}_iqr']
            elif self.method == 'zscore' and f'{col}_zscore' in self.bounds_:
                lower, upper = self.bounds_[f'{col}_zscore']
            elif self.method == 'both' and f'{col}_iqr' in self.bounds_:
                lower, upper = self.bounds_[f'{col}_iqr']
            else:
                continue
            
            outliers = (data[col] < lower) | (data[col] > upper)
            outlier_counts[col] = outliers.sum()
            
            if self.action == 'clip':
                data[col] = data[col].clip(lower=lower, upper=upper)
            elif self.action == 'remove':
                data = data[~outliers]
            elif self.action == 'mark':
                data[f'{col}_outlier'] = outliers.astype(int)
        
        self.outlier_stats_ = outlier_counts
        return data
    
    def report(self) -> Dict[str, Any]:
        """Generate outlier report."""
        return {
            'preprocessor': 'OutlierHandler',
            'method': self.method,
            'threshold': self.threshold,
            'action': self.action,
            'outlier_statistics': self.outlier_stats_,
            'bounds': {k: str(v) for k, v in self.bounds_.items()},
            'is_fitted': self.is_fitted
        }

