"""
Golden Profiles - Reference curves for fermentation phases
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class GoldenProfiles:
    """Generate and manage golden profiles for different strains and styles."""
    
    def __init__(self):
        self.profiles = {}
    
    def create_profile(self, strain: str, style: str, 
                      duration_hours: int = 168,
                      peak_co2: float = 2000.0) -> pd.DataFrame:
        """Create a golden profile curve for a strain/style combination."""
        # Create time points (every 5 minutes)
        time_points = np.arange(0, duration_hours * 60, 5)
        n_points = len(time_points)
        
        # Generate CO2 curve (typical fermentation pattern)
        # Lag phase: 0-10% of time, low CO2
        # Exponential: 10-40% of time, rapid increase
        # Stationary: 40-70% of time, peak CO2
        # Decline: 70-100% of time, gradual decrease
        
        lag_end = int(n_points * 0.1)
        exp_end = int(n_points * 0.4)
        stat_end = int(n_points * 0.7)
        
        co2_curve = np.zeros(n_points)
        
        # Lag phase: very low CO2
        co2_curve[:lag_end] = np.linspace(0, peak_co2 * 0.05, lag_end)
        
        # Exponential phase: rapid growth
        exp_points = exp_end - lag_end
        co2_curve[lag_end:exp_end] = np.linspace(
            peak_co2 * 0.05, peak_co2 * 0.8, exp_points
        )
        
        # Stationary phase: peak and slight variation
        stat_points = stat_end - exp_end
        co2_curve[exp_end:stat_end] = peak_co2 * (0.8 + 0.2 * np.sin(
            np.linspace(0, np.pi, stat_points)
        ))
        
        # Decline phase: gradual decrease
        decline_points = n_points - stat_end
        co2_curve[stat_end:] = np.linspace(
            peak_co2 * 0.9, peak_co2 * 0.3, decline_points
        )
        
        # Generate DO curve (decreases over time)
        do_initial = 20.0
        do_final = 0.5
        do_curve = np.linspace(do_initial, do_final, n_points)
        do_curve += np.random.normal(0, 0.5, n_points)  # Add noise
        do_curve = np.clip(do_curve, 0, 21)
        
        # Generate temperature curve (slight increase then stable)
        temp_initial = 18.0
        temp_peak = 22.0
        temp_curve = np.zeros(n_points)
        temp_curve[:exp_end] = np.linspace(temp_initial, temp_peak, exp_end)
        temp_curve[exp_end:] = temp_peak
        
        # Generate pressure curve (increases with CO2)
        pressure_initial = 1.0  # bar
        pressure_peak = 1.5  # bar
        pressure_curve = pressure_initial + (co2_curve / peak_co2) * (pressure_peak - pressure_initial)
        
        # Create DataFrame
        profile = pd.DataFrame({
            'time_minutes': time_points,
            'co2_lpm': co2_curve,
            'do_ppm': do_curve,
            'temp_c': temp_curve,
            'pressure_bar': pressure_curve,
            'phase': self._assign_phases(n_points, lag_end, exp_end, stat_end)
        })
        
        return profile
    
    def _assign_phases(self, n_points: int, lag_end: int, 
                      exp_end: int, stat_end: int) -> List[str]:
        """Assign phase labels to time points."""
        phases = []
        for i in range(n_points):
            if i < lag_end:
                phases.append('lag')
            elif i < exp_end:
                phases.append('exponential')
            elif i < stat_end:
                phases.append('stationary')
            else:
                phases.append('decline')
        return phases
    
    def get_profile(self, strain: str, style: str) -> Optional[pd.DataFrame]:
        """Get golden profile for a strain/style combination."""
        key = (strain, style)
        if key in self.profiles:
            return self.profiles[key]
        return None
    
    def add_profile(self, strain: str, style: str, profile: pd.DataFrame):
        """Add a golden profile."""
        key = (strain, style)
        self.profiles[key] = profile
    
    def generate_default_profiles(self):
        """Generate default golden profiles for common combinations."""
        # Default strain/style combinations
        combinations = [
            ('default_strain', 'default_style'),
            ('ale_strain', 'ipa'),
            ('lager_strain', 'pilsner'),
        ]
        
        for strain, style in combinations:
            profile = self.create_profile(strain, style)
            self.add_profile(strain, style, profile)
    
    def save_profiles(self, filepath: str):
        """Save all profiles to a file."""
        import json
        profiles_dict = {}
        for (strain, style), profile in self.profiles.items():
            profiles_dict[f"{strain}_{style}"] = profile.to_dict('records')
        
        with open(filepath, 'w') as f:
            json.dump(profiles_dict, f, indent=2, default=str)
    
    def load_profiles(self, filepath: str):
        """Load profiles from a file."""
        import json
        with open(filepath, 'r') as f:
            profiles_dict = json.load(f)
        
        for key, records in profiles_dict.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                strain, style = parts[0], parts[1]
                profile = pd.DataFrame(records)
                self.add_profile(strain, style, profile)

