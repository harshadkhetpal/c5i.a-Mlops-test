"""
Unit tests for preprocessing modules
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    MissingHandler, OutlierHandler, Normalizer,
    Resampler, Aligner, SchemaMapper, GoldenProfiles
)

class TestSchemaMapper(unittest.TestCase):
    """Test SchemaMapper functionality."""
    
    def test_map_to_case_study_schema(self):
        """Test mapping to case study schema."""
        data = pd.DataFrame({
            'timestamp_index': pd.date_range('2025-01-01', periods=10, freq='5min'),
            'co2_ppm': np.random.rand(10) * 2000,
            'o2_pct': np.random.rand(10) * 20,
            'pressure_kpa': np.random.rand(10) * 50 + 100,
            'process_temp_c': np.random.rand(10) * 10 + 20
        })
        
        mapped = SchemaMapper.map_to_case_study_schema(data)
        
        self.assertIn('co2_lpm', mapped.columns)
        self.assertIn('do_ppm', mapped.columns)
        self.assertIn('pressure_bar', mapped.columns)
        self.assertIn('valve_state', mapped.columns)
        self.assertIn('agitator_rpm', mapped.columns)
    
    def test_synthetic_columns(self):
        """Test synthetic column generation."""
        data = pd.DataFrame({
            'timestamp_index': pd.date_range('2025-01-01', periods=5, freq='5min'),
            'co2_ppm': [100, 200, 300, 400, 500]
        })
        
        mapped = SchemaMapper.map_to_case_study_schema(data)
        
        self.assertIn('valve_state', mapped.columns)
        self.assertIn('agitator_rpm', mapped.columns)
        self.assertIn('pitch_time', mapped.columns)
        self.assertIn('OG', mapped.columns)
        self.assertIn('target_attenuation', mapped.columns)

class TestMissingHandler(unittest.TestCase):
    """Test MissingHandler."""
    
    def test_missing_value_handling(self):
        """Test missing value interpolation."""
        data = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 5, np.nan, 7]
        })
        
        handler = MissingHandler(method='interpolate')
        result = handler.fit_transform(data)
        
        self.assertFalse(result['value'].isnull().any())

class TestOutlierHandler(unittest.TestCase):
    """Test OutlierHandler."""
    
    def test_outlier_clipping(self):
        """Test outlier clipping."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]  # 100 is outlier
        })
        
        handler = OutlierHandler(method='iqr', action='clip')
        result = handler.fit_transform(data)
        
        # Outlier should be clipped
        self.assertLess(result['value'].max(), 100)

class TestNormalizer(unittest.TestCase):
    """Test Normalizer."""
    
    def test_tank_level_normalization(self):
        """Test tank-level normalization."""
        data = pd.DataFrame({
            'tank_id': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 100, 200]
        })
        
        normalizer = Normalizer(method='standard', group_by='tank_id')
        result = normalizer.fit_transform(data)
        
        # Each tank should be normalized separately
        self.assertIn('value', result.columns)

class TestGoldenProfiles(unittest.TestCase):
    """Test GoldenProfiles."""
    
    def test_profile_generation(self):
        """Test golden profile generation."""
        profiles = GoldenProfiles()
        profile = profiles.create_profile('test_strain', 'test_style')
        
        self.assertIn('co2_lpm', profile.columns)
        self.assertIn('do_ppm', profile.columns)
        self.assertIn('phase', profile.columns)
        self.assertEqual(len(profile), 2016)  # 168 hours * 12 samples/hour

if __name__ == '__main__':
    unittest.main()

