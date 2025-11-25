import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_preprocessor import BasePreprocessor

class Aligner(BasePreprocessor):
    """Align time-series to golden profiles."""
    
    def __init__(self, golden_profiles: Optional[pd.DataFrame] = None, 
                 strain_col: str = 'strain', style_col: str = 'style'):
        super().__init__()
        self.golden_profiles = golden_profiles
        self.strain_col = strain_col
        self.style_col = style_col
        self.profiles_ = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'Aligner':
        """Load golden profiles or create from data."""
        if self.golden_profiles is not None:
            # Organize golden profiles by strain and style
            for _, row in self.golden_profiles.iterrows():
                key = (row[self.strain_col], row[self.style_col])
                self.profiles_[key] = row.to_dict()
        else:
            # Create simple golden profiles from data statistics
            if self.strain_col in data.columns and self.style_col in data.columns:
                for strain in data[self.strain_col].unique():
                    for style in data[data[self.strain_col] == strain][self.style_col].unique():
                        subset = data[(data[self.strain_col] == strain) & 
                                     (data[self.style_col] == style)]
                        key = (strain, style)
                        self.profiles_[key] = {
                            'mean_co2': subset.get('co2_ppm', pd.Series()).mean() if 'co2_ppm' in subset.columns else 0,
                            'max_co2': subset.get('co2_ppm', pd.Series()).max() if 'co2_ppm' in subset.columns else 0,
                        }
            else:
                # Create global profile
                self.profiles_[('default', 'default')] = {
                    'mean_co2': data.get('co2_ppm', pd.Series()).mean() if 'co2_ppm' in data.columns else 0,
                    'max_co2': data.get('co2_ppm', pd.Series()).max() if 'co2_ppm' in data.columns else 0,
                }
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, 
                  target_col: str = 'co2_ppm', **kwargs) -> pd.DataFrame:
        """Align data to golden profiles."""
        data = data.copy()
        
        # Add phase information based on golden profile alignment
        if 'phase' not in data.columns:
            data['phase'] = self._detect_phase(data, target_col)
        
        return data
    
    def _detect_phase(self, data: pd.DataFrame, target_col: str) -> pd.Series:
        """Detect fermentation phase based on golden profile."""
        phases = []
        
        for idx, row in data.iterrows():
            if self.strain_col in row and self.style_col in row:
                key = (row[self.strain_col], row[self.style_col])
            else:
                key = ('default', 'default')
            
            if key in self.profiles_:
                current_val = row.get(target_col, 0)
                profile = self.profiles_[key]
                max_val = profile.get('max_co2', current_val) if current_val > 0 else 1
                
                if max_val > 0:
                    ratio = current_val / max_val
                    if ratio < 0.1:
                        phase = 'lag'
                    elif ratio < 0.5:
                        phase = 'exponential'
                    elif ratio < 0.9:
                        phase = 'stationary'
                    else:
                        phase = 'decline'
                else:
                    phase = 'unknown'
            else:
                # Simple phase detection based on value
                current_val = row.get(target_col, 0)
                if current_val < 500:
                    phase = 'lag'
                elif current_val < 1000:
                    phase = 'exponential'
                elif current_val < 1500:
                    phase = 'stationary'
                else:
                    phase = 'decline'
            
            phases.append(phase)
        
        return pd.Series(phases, index=data.index)
    
    def report(self) -> Dict[str, Any]:
        """Generate alignment report."""
        return {
            'preprocessor': 'Aligner',
            'profiles_loaded': len(self.profiles_),
            'is_fitted': self.is_fitted
        }

