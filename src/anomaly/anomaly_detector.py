import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class AnomalyDetector:
    """Detect anomalies in fermentation data."""
    
    def __init__(self):
        self.anomalies_ = []
        self.anomaly_timeline_ = []
    
    def detect_stuck_fermentation(self, data: pd.DataFrame,
                                  co2_col: str = 'co2_ppm',
                                  time_col: str = 'timestamp_index',
                                  window: int = 12,
                                  threshold: float = 0.01) -> pd.DataFrame:
        """Detect stuck fermentation (CO2 stops increasing)."""
        data = data.copy()
        data = data.sort_values(time_col)
        
        if co2_col not in data.columns:
            return pd.DataFrame()
        
        # Compute CO2 rate of change
        data['co2_rate'] = data[co2_col].diff()
        data['co2_rolling_mean'] = data['co2_rate'].rolling(window=window, min_periods=1).mean()
        
        # Detect stuck fermentation: rate near zero when it should be positive
        stuck_mask = (data['co2_rolling_mean'] < threshold) & (data[co2_col] < data[co2_col].quantile(0.7))
        
        anomalies = data[stuck_mask].copy()
        anomalies['anomaly_type'] = 'stuck_fermentation'
        anomalies['severity'] = 'high'
        
        self.anomalies_.extend(anomalies.to_dict('records'))
        
        return anomalies
    
    def detect_oxidation_risk(self, data: pd.DataFrame,
                             do_col: str = 'o2_pct',
                             time_col: str = 'timestamp_index',
                             threshold: float = 0.5) -> pd.DataFrame:
        """Detect oxidation risks (DO spikes unexpectedly)."""
        data = data.copy()
        data = data.sort_values(time_col)
        
        if do_col not in data.columns:
            return pd.DataFrame()
        
        # Compute DO change rate
        data['do_change'] = data[do_col].diff()
        data['do_change_abs'] = abs(data['do_change'])
        
        # Detect sudden increases in DO
        mean_change = data['do_change_abs'].mean()
        std_change = data['do_change_abs'].std()
        
        oxidation_mask = data['do_change'] > (mean_change + threshold * std_change)
        
        anomalies = data[oxidation_mask].copy()
        anomalies['anomaly_type'] = 'oxidation_risk'
        anomalies['severity'] = 'medium'
        
        self.anomalies_.extend(anomalies.to_dict('records'))
        
        return anomalies
    
    def detect_pressure_anomalies(self, data: pd.DataFrame,
                                 pressure_col: str = 'pressure_kpa',
                                 time_col: str = 'timestamp_index',
                                 max_pressure: float = 150.0,
                                 min_pressure: float = 80.0) -> pd.DataFrame:
        """Detect pressure anomalies."""
        data = data.copy()
        
        if pressure_col not in data.columns:
            return pd.DataFrame()
        
        # Detect pressure outside safe range
        high_pressure = data[data[pressure_col] > max_pressure].copy()
        low_pressure = data[data[pressure_col] < min_pressure].copy()
        
        anomalies = pd.concat([high_pressure, low_pressure], ignore_index=True)
        
        if len(anomalies) > 0:
            anomalies['anomaly_type'] = anomalies.apply(
                lambda row: 'high_pressure' if row[pressure_col] > max_pressure else 'low_pressure',
                axis=1
            )
            anomalies['severity'] = anomalies['anomaly_type'].apply(
                lambda x: 'high' if x == 'high_pressure' else 'medium'
            )
            
            self.anomalies_.extend(anomalies.to_dict('records'))
        
        return anomalies
    
    def detect_abnormal_co2_activity(self, data: pd.DataFrame,
                                    co2_col: str = 'co2_ppm',
                                    time_col: str = 'timestamp_index',
                                    golden_profile: Optional[pd.DataFrame] = None,
                                    threshold: float = 0.3) -> pd.DataFrame:
        """Detect abnormal CO2 activity compared to golden profile."""
        data = data.copy()
        data = data.sort_values(time_col)
        
        if co2_col not in data.columns:
            return pd.DataFrame()
        
        anomalies = pd.DataFrame()
        
        if golden_profile is not None and co2_col in golden_profile.columns:
            # Compare to golden profile
            try:
                from src.analytics.numpy_operations import NumPyOperations
                similarity = NumPyOperations.compare_to_golden(data, golden_profile, co2_col)
            except ImportError:
                # Fallback if import fails
                similarity = 0.5
            
            if similarity < (1 - threshold):
                # Mark as abnormal
                anomalies = data.copy()
                anomalies['anomaly_type'] = 'abnormal_co2_pattern'
                anomalies['severity'] = 'medium'
                anomalies['similarity_score'] = similarity
        else:
            # Use statistical method: detect rapid changes
            data['co2_rate'] = data[co2_col].diff()
            mean_rate = data['co2_rate'].mean()
            std_rate = data['co2_rate'].std()
            
            # Detect rapid increases or decreases
            rapid_change = abs(data['co2_rate']) > (mean_rate + 3 * std_rate)
            
            anomalies = data[rapid_change].copy()
            anomalies['anomaly_type'] = 'abnormal_co2_activity'
            anomalies['severity'] = 'medium'
        
        if len(anomalies) > 0:
            self.anomalies_.extend(anomalies.to_dict('records'))
        
        return anomalies
    
    def detect_all(self, data: pd.DataFrame,
                  co2_col: str = 'co2_ppm',
                  do_col: str = 'o2_pct',
                  pressure_col: str = 'pressure_kpa',
                  time_col: str = 'timestamp_index',
                  golden_profile: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Detect all types of anomalies."""
        self.anomalies_ = []
        
        # Detect all anomaly types
        stuck = self.detect_stuck_fermentation(data, co2_col, time_col)
        oxidation = self.detect_oxidation_risk(data, do_col, time_col)
        pressure = self.detect_pressure_anomalies(data, pressure_col, time_col)
        abnormal_co2 = self.detect_abnormal_co2_activity(data, co2_col, time_col, golden_profile)
        
        # Combine all anomalies
        all_anomalies = pd.concat([stuck, oxidation, pressure, abnormal_co2], ignore_index=True)
        
        # Remove duplicates
        if len(all_anomalies) > 0 and time_col in all_anomalies.columns:
            all_anomalies = all_anomalies.drop_duplicates(subset=[time_col])
        
        # Create timeline
        if len(all_anomalies) > 0:
            self.anomaly_timeline_ = all_anomalies.sort_values(time_col).to_dict('records')
        
        return all_anomalies
    
    def get_anomaly_timeline(self) -> List[Dict[str, Any]]:
        """Get anomaly timeline."""
        return self.anomaly_timeline_
    
    def generate_anomaly_report(self) -> str:
        """Generate anomaly report."""
        if not self.anomalies_:
            return "No anomalies detected."
        
        report = "=== ANOMALY DETECTION REPORT ===\n\n"
        report += f"Total anomalies detected: {len(self.anomalies_)}\n\n"
        
        # Group by type
        anomaly_df = pd.DataFrame(self.anomalies_)
        if 'anomaly_type' in anomaly_df.columns:
            type_counts = anomaly_df['anomaly_type'].value_counts()
            report += "Anomalies by type:\n"
            for anomaly_type, count in type_counts.items():
                report += f"  - {anomaly_type}: {count}\n"
        
        report += "\n"
        return report

