import pandas as pd
import numpy as np
from typing import List, Tuple

class FeatureEngineering:
    """Feature engineering for fermentation data."""
    
    @staticmethod
    def create_polynomial_features(data: pd.DataFrame, columns: List[str], 
                                   degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        data = data.copy()
        
        for col in columns:
            if col in data.columns:
                for d in range(2, degree + 1):
                    data[f'{col}_poly_{d}'] = data[col] ** d
        
        return data
    
    @staticmethod
    def create_interaction_features(data: pd.DataFrame, 
                                    pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features."""
        data = data.copy()
        
        for col1, col2 in pairs:
            if col1 in data.columns and col2 in data.columns:
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        
        return data
    
    @staticmethod
    def create_lag_features(data: pd.DataFrame, columns: List[str], 
                           lags: List[int] = [5, 15, 60]) -> pd.DataFrame:
        """Create lag features in minutes."""
        data = data.copy()
        time_col = 'timestamp_index' if 'timestamp_index' in data.columns else data.columns[0]
        data = data.sort_values(time_col)
        
        # Assuming 5-minute intervals (adjust if different)
        lag_steps = [l // 5 for l in lags]
        
        for col in columns:
            if col in data.columns:
                for lag in lag_steps:
                    if lag > 0:
                        data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return data
    
    @staticmethod
    def create_rolling_features(data: pd.DataFrame, columns: List[str],
                                windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """Create rolling statistical features."""
        data = data.copy()
        time_col = 'timestamp_index' if 'timestamp_index' in data.columns else data.columns[0]
        data = data.sort_values(time_col)
        
        for col in columns:
            if col in data.columns:
                for window in windows:
                    data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window, min_periods=1).mean()
                    data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window, min_periods=1).std()
                    data[f'{col}_rolling_skew_{window}'] = data[col].rolling(window=window, min_periods=1).skew()
                    data[f'{col}_rolling_kurt_{window}'] = data[col].rolling(window=window, min_periods=1).apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0.0, raw=False)
        
        return data
    
    @staticmethod
    def create_phase_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create phase binning features."""
        data = data.copy()
        
        # Phase encoding
        phase_map = {'lag': 0, 'exponential': 1, 'stationary': 2, 'decline': 3, 'unknown': -1}
        if 'phase' in data.columns:
            data['phase_encoded'] = data['phase'].map(phase_map).fillna(-1)
        
        return data
    
    @staticmethod
    def create_temporal_features(data: pd.DataFrame, 
                                 time_col: str = 'timestamp_index') -> pd.DataFrame:
        """Create temporal features from timestamp."""
        data = data.copy()
        
        if time_col in data.columns:
            data[time_col] = pd.to_datetime(data[time_col])
            data['hour'] = data[time_col].dt.hour
            data['day_of_week'] = data[time_col].dt.dayofweek
            data['day_of_month'] = data[time_col].dt.day
            data['month'] = data[time_col].dt.month
            
            # Cyclical encoding
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    @staticmethod
    def create_all_features(data: pd.DataFrame, 
                           co2_col: str = 'co2_lpm',
                           temp_col: str = 'temp_c',
                           pressure_col: str = 'pressure_bar',
                           do_col: str = 'do_ppm') -> pd.DataFrame:
        """Create all features."""
        # Temporal features
        data = FeatureEngineering.create_temporal_features(data)
        
        # Add valve_state and agitator_rpm features if available
        if 'valve_state' in data.columns:
            data['valve_state_binary'] = (data['valve_state'] > 0).astype(int)
            data['valve_state_change'] = data['valve_state'].diff()
        
        if 'agitator_rpm' in data.columns:
            data['agitator_rpm_normalized'] = data['agitator_rpm'] / (data['agitator_rpm'].max() + 1e-10)
            data['agitator_rpm_change'] = data['agitator_rpm'].diff()
        
        # Add attenuation features if OG available
        if 'OG' in data.columns and co2_col in data.columns:
            # Estimate current gravity from CO2 production
            initial_co2 = data[co2_col].iloc[0] if len(data) > 0 else 0
            current_co2 = data[co2_col]
            # Approximate attenuation percentage
            data['estimated_attenuation'] = ((current_co2 - initial_co2) / (initial_co2 + 1e-10)) * 100
            if 'target_attenuation' in data.columns:
                data['attenuation_deviation'] = data['estimated_attenuation'] - data['target_attenuation']
        
        # Polynomial features
        poly_cols = [c for c in [co2_col, temp_col] if c in data.columns]
        if poly_cols:
            data = FeatureEngineering.create_polynomial_features(data, poly_cols, degree=2)
        
        # Interaction features
        interaction_pairs = []
        if co2_col in data.columns and temp_col in data.columns:
            interaction_pairs.append((co2_col, temp_col))
        if co2_col in data.columns and pressure_col in data.columns:
            interaction_pairs.append((co2_col, pressure_col))
        if 'valve_state' in data.columns and co2_col in data.columns:
            interaction_pairs.append(('valve_state', co2_col))
        if 'agitator_rpm' in data.columns and co2_col in data.columns:
            interaction_pairs.append(('agitator_rpm', co2_col))
        if interaction_pairs:
            data = FeatureEngineering.create_interaction_features(data, interaction_pairs)
        
        # Lag features
        lag_cols = [c for c in [co2_col, do_col, temp_col, 'valve_state', 'agitator_rpm'] 
                   if c in data.columns]
        if lag_cols:
            data = FeatureEngineering.create_lag_features(data, lag_cols, lags=[5, 15, 60])
        
        # Rolling features
        rolling_cols = [c for c in [co2_col, do_col, 'valve_state', 'agitator_rpm'] 
                       if c in data.columns]
        if rolling_cols:
            data = FeatureEngineering.create_rolling_features(data, rolling_cols, windows=[6, 12, 24])
        
        # Phase features
        data = FeatureEngineering.create_phase_features(data)
        
        return data

