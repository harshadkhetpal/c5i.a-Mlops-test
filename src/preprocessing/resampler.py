import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_preprocessor import BasePreprocessor

class Resampler(BasePreprocessor):
    """Resample time-series to uniform intervals."""
    
    def __init__(self, freq: str = '5T', method: str = 'mean', 
                 time_col: str = 'timestamp_index'):
        super().__init__()
        self.freq = freq  # '5T' for 5 minutes
        self.method = method  # 'mean', 'median', 'interpolate'
        self.time_col = time_col
        self.original_freq_ = None
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'Resampler':
        """Record original frequency."""
        data = data.copy()
        if self.time_col in data.columns:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
            data = data.set_index(self.time_col)
            
            # Calculate most common frequency
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                self.original_freq_ = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, group_by: str = None, **kwargs) -> pd.DataFrame:
        """Resample to uniform intervals."""
        data = data.copy()
        
        if self.time_col not in data.columns:
            return data
        
        data[self.time_col] = pd.to_datetime(data[self.time_col])
        
        if group_by and group_by in data.columns:
            resampled_dfs = []
            
            for group_id in data[group_by].unique():
                group_data = data[data[group_by] == group_id].copy()
                group_data = group_data.set_index(self.time_col)
                
                resampled = self._resample_group(group_data)
                resampled[group_by] = group_id
                resampled = resampled.reset_index()
                resampled_dfs.append(resampled)
            
            result = pd.concat(resampled_dfs, ignore_index=True)
        else:
            data = data.set_index(self.time_col)
            result = self._resample_group(data)
            result = result.reset_index()
        
        return result
    
    def _resample_group(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample a single group."""
        # Only resample numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in data.columns if c not in numeric_cols]
        
        if len(numeric_cols) == 0:
            return data
        
        numeric_data = data[numeric_cols]
        
        if self.method == 'mean':
            resampled = numeric_data.resample(self.freq).mean()
        elif self.method == 'median':
            resampled = numeric_data.resample(self.freq).median()
        elif self.method == 'interpolate':
            resampled = numeric_data.resample(self.freq).mean().interpolate()
        else:
            resampled = numeric_data.resample(self.freq).mean()
        
        # Add back non-numeric columns (forward fill)
        if non_numeric_cols:
            non_numeric_data = data[non_numeric_cols].resample(self.freq).first()
            resampled = pd.concat([resampled, non_numeric_data], axis=1)
        
        return resampled
    
    def report(self) -> Dict[str, Any]:
        """Generate resampling report."""
        return {
            'preprocessor': 'Resampler',
            'target_frequency': self.freq,
            'method': self.method,
            'original_frequency': str(self.original_freq_),
            'is_fitted': self.is_fitted
        }

