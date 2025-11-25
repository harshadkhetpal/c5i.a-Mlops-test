import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base_preprocessor import BasePreprocessor

class MissingHandler(BasePreprocessor):
    """Handle missing values using interpolation and forward fill."""
    
    def __init__(self, method: str = 'interpolate', limit: int = 3):
        super().__init__()
        self.method = method  # 'interpolate', 'forward_fill', 'both'
        self.limit = limit
        self.missing_stats_ = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MissingHandler':
        """Record missing value statistics."""
        self.missing_stats_ = {
            'total_missing': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'columns_with_missing': data.columns[data.isnull().any()].tolist()
        }
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Handle missing values."""
        data = data.copy()
        
        if self.method in ['interpolate', 'both']:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit=self.limit)
        
        if self.method in ['forward_fill', 'both']:
            data = data.ffill(limit=self.limit)
        
        # Fill remaining with backward fill
        data = data.bfill()
        
        return data
    
    def report(self) -> Dict[str, Any]:
        """Generate missing value report."""
        return {
            'preprocessor': 'MissingHandler',
            'method': self.method,
            'missing_statistics': self.missing_stats_,
            'is_fitted': self.is_fitted
        }

