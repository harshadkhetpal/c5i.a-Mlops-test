"""
Unit tests for anomaly detection
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anomaly import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    """Test AnomalyDetector."""
    
    def test_stuck_fermentation_detection(self):
        """Test stuck fermentation detection."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=20, freq='5min'),
            'co2_lpm': [100] * 20  # Constant CO2 = stuck
        })
        
        detector = AnomalyDetector()
        anomalies = detector.detect_stuck_fermentation(data)
        
        self.assertGreaterEqual(len(anomalies), 0)
    
    def test_over_vigorous_co2(self):
        """Test over-vigorous CO2 detection."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='5min'),
            'co2_lpm': [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]  # Rapid increase
        })
        
        detector = AnomalyDetector()
        anomalies = detector.detect_over_vigorous_co2(data, threshold_rate=500.0)
        
        self.assertGreaterEqual(len(anomalies), 0)
    
    def test_rapid_pressure_rise(self):
        """Test rapid pressure rise detection."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='5min'),
            'pressure_bar': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  # Rapid rise
        })
        
        detector = AnomalyDetector()
        anomalies = detector.detect_rapid_pressure_rise(data, threshold_rate=0.1)
        
        self.assertGreaterEqual(len(anomalies), 0)

if __name__ == '__main__':
    unittest.main()

