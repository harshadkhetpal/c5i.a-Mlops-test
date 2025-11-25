import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class PandasAnalytics:
    """Advanced pandas analytics for fermentation data."""
    
    @staticmethod
    def compute_batch_metrics(data: pd.DataFrame, batch_id: str = 'batch_id',
                             time_col: str = 'timestamp_index',
                             co2_col: str = 'co2_ppm',
                             do_col: str = 'o2_pct',
                             pressure_col: str = 'pressure_kpa') -> pd.DataFrame:
        """Compute batch-level metrics."""
        metrics = []
        
        # Create batch_id if it doesn't exist
        if batch_id not in data.columns:
            # Create synthetic batches based on time windows
            data = data.copy()
            data[batch_id] = (pd.to_datetime(data[time_col]).dt.date.astype(str))
        
        for bid in data[batch_id].unique():
            batch_data = data[data[batch_id] == bid].copy()
            batch_data = batch_data.sort_values(time_col)
            
            if len(batch_data) == 0:
                continue
            
            # Peak CO2 and time to peak
            if co2_col in batch_data.columns:
                peak_co2 = batch_data[co2_col].max()
                peak_idx = batch_data[co2_col].idxmax()
                time_to_peak = batch_data.loc[peak_idx, time_col]
                start_time = pd.to_datetime(batch_data[time_col].min())
                time_to_peak_dt = pd.to_datetime(time_to_peak)
                time_to_peak_minutes = (time_to_peak_dt - start_time).total_seconds() / 60
            else:
                peak_co2 = None
                time_to_peak_minutes = None
            
            # DO half-life
            do_half_life = None
            if do_col in batch_data.columns and len(batch_data) > 0:
                initial_do = batch_data[do_col].iloc[0]
                if initial_do > 0:
                    half_do = initial_do / 2
                    below_half = batch_data[batch_data[do_col] <= half_do]
                    if len(below_half) > 0:
                        do_half_life_idx = below_half.index[0]
                        start_time = pd.to_datetime(batch_data[time_col].min())
                        do_half_life_dt = pd.to_datetime(batch_data.loc[do_half_life_idx, time_col])
                        do_half_life = (do_half_life_dt - start_time).total_seconds() / 60
            
            # Pressure growth
            pressure_growth = None
            if pressure_col in batch_data.columns and len(batch_data) > 0:
                initial_pressure = batch_data[pressure_col].iloc[0]
                final_pressure = batch_data[pressure_col].iloc[-1]
                pressure_growth = final_pressure - initial_pressure
            
            # Duration
            start_time = pd.to_datetime(batch_data[time_col].min())
            end_time = pd.to_datetime(batch_data[time_col].max())
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            metrics.append({
                'batch_id': bid,
                'peak_co2_ppm': peak_co2,
                'time_to_peak_minutes': time_to_peak_minutes,
                'do_half_life_minutes': do_half_life,
                'pressure_growth_kpa': pressure_growth,
                'duration_hours': duration_hours,
                'num_samples': len(batch_data)
            })
        
        return pd.DataFrame(metrics)
    
    @staticmethod
    def compute_rolling_statistics(data: pd.DataFrame, window: int = 12, 
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute rolling statistics."""
        if columns is None:
            columns = ['co2_ppm', 'o2_pct', 'process_temp_c', 'pressure_kpa']
            columns = [c for c in columns if c in data.columns]
        
        data = data.copy()
        time_col = 'timestamp_index' if 'timestamp_index' in data.columns else data.columns[0]
        data = data.sort_values(time_col)
        
        for col in columns:
            if col in data.columns:
                data[f'{col}_rolling_mean'] = data[col].rolling(window=window, min_periods=1).mean()
                data[f'{col}_rolling_std'] = data[col].rolling(window=window, min_periods=1).std()
                # Use apply for kurtosis as it's not directly available
                data[f'{col}_rolling_kurtosis'] = data[col].rolling(window=window, min_periods=1).apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0.0, raw=False)
                data[f'{col}_rolling_cv'] = data[f'{col}_rolling_std'] / (data[f'{col}_rolling_mean'] + 1e-10)
        
        return data
    
    @staticmethod
    def create_pivot_comparison(data: pd.DataFrame, 
                                index: str = 'batch_id',
                                columns: Optional[str] = None,
                                values: str = 'peak_co2_ppm',
                                aggfunc: str = 'mean') -> pd.DataFrame:
        """Create pivot table for batch comparison."""
        if columns is None:
            return pd.DataFrame({values: data.groupby(index)[values].agg(aggfunc)})
        return pd.pivot_table(data, index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    @staticmethod
    def compute_attenuation_slope(data: pd.DataFrame, 
                                  batch_id: str = 'batch_id',
                                  co2_col: str = 'co2_ppm',
                                  time_col: str = 'timestamp_index') -> pd.DataFrame:
        """Compute attenuation slope for batches."""
        slopes = []
        
        if batch_id not in data.columns:
            data = data.copy()
            data[batch_id] = (pd.to_datetime(data[time_col]).dt.date.astype(str))
        
        for bid in data[batch_id].unique():
            batch_data = data[data[batch_id] == bid].copy()
            batch_data = batch_data.sort_values(time_col)
            
            if len(batch_data) < 2 or co2_col not in batch_data.columns:
                continue
            
            # Compute slope using linear regression
            x = np.arange(len(batch_data))
            y = batch_data[co2_col].values
            
            if len(y) > 1 and np.std(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append({
                    'batch_id': bid,
                    'attenuation_slope': slope,
                    'initial_co2': y[0],
                    'final_co2': y[-1]
                })
        
        return pd.DataFrame(slopes)

