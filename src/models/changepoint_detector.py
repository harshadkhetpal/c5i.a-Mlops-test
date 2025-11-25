import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy import stats

class ChangepointDetector:
    """Detect changepoints in time-series data."""
    
    def __init__(self, method: str = 'cusum'):
        self.method = method  # 'cusum', 'pelt', 'window'
        self.changepoints_ = []
    
    def detect(self, data: pd.DataFrame, 
               column: str = 'co2_ppm',
               time_col: str = 'timestamp_index',
               threshold: float = 3.0) -> List[int]:
        """Detect changepoints in the time series."""
        if column not in data.columns:
            return []
        
        series = data[column].values
        self.changepoints_ = []
        
        if self.method == 'cusum':
            self.changepoints_ = self._cusum_detection(series, threshold)
        elif self.method == 'window':
            self.changepoints_ = self._window_detection(series, threshold)
        else:
            self.changepoints_ = self._simple_detection(series, threshold)
        
        return self.changepoints_
    
    def _cusum_detection(self, series: np.ndarray, threshold: float) -> List[int]:
        """CUSUM-based changepoint detection."""
        changepoints = []
        
        # Compute cumulative sum of deviations from mean
        mean = np.mean(series)
        cumsum = np.cumsum(series - mean)
        
        # Find points where cumulative sum exceeds threshold
        for i in range(1, len(cumsum)):
            if abs(cumsum[i]) > threshold * np.std(series):
                changepoints.append(i)
        
        return changepoints
    
    def _window_detection(self, series: np.ndarray, threshold: float) -> List[int]:
        """Window-based changepoint detection."""
        changepoints = []
        window_size = min(50, len(series) // 10)
        
        for i in range(window_size, len(series) - window_size):
            window_before = series[i - window_size:i]
            window_after = series[i:i + window_size]
            
            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            
            # Statistical test for difference
            if abs(mean_after - mean_before) > threshold * np.std(series):
                changepoints.append(i)
        
        return changepoints
    
    def _simple_detection(self, series: np.ndarray, threshold: float) -> List[int]:
        """Simple derivative-based detection."""
        changepoints = []
        diff = np.diff(series)
        threshold_val = threshold * np.std(diff)
        
        for i in range(len(diff)):
            if abs(diff[i]) > threshold_val:
                changepoints.append(i + 1)
        
        return changepoints
    
    def detect_phase_boundaries(self, data: pd.DataFrame,
                               column: str = 'co2_ppm',
                               time_col: str = 'timestamp_index') -> Dict[str, Any]:
        """Detect phase boundaries."""
        changepoints = self.detect(data, column, time_col)
        
        if len(changepoints) == 0:
            return {'boundaries': [], 'phases': []}
        
        # Map changepoints to phases
        phases = []
        boundaries = []
        
        for i, cp in enumerate(changepoints):
            if cp < len(data):
                boundaries.append({
                    'index': cp,
                    'timestamp': data.iloc[cp][time_col] if time_col in data.columns else None,
                    'value': data.iloc[cp][column] if column in data.columns else None
                })
        
        return {
            'boundaries': boundaries,
            'num_changepoints': len(changepoints)
        }
    
    def get_changepoints(self) -> List[int]:
        """Get detected changepoints."""
        return self.changepoints_

