"""
Unit tests for model modules
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PhasePredictor, TemporalForest

class TestPhasePredictor(unittest.TestCase):
    """Test PhasePredictor."""
    
    def test_model_creation(self):
        """Test model initialization."""
        predictor = PhasePredictor()
        self.assertIsNotNone(predictor.model)
        self.assertFalse(predictor.is_trained)
    
    def test_feature_preparation(self):
        """Test feature preparation."""
        data = pd.DataFrame({
            'co2_lpm': [100, 200, 300],
            'do_ppm': [20, 19, 18],
            'temp_c': [20, 21, 22]
        })
        
        predictor = PhasePredictor()
        X = predictor.prepare_features(data)
        
        self.assertGreater(X.shape[1], 0)

class TestTemporalForest(unittest.TestCase):
    """Test TemporalForest."""
    
    def test_model_creation(self):
        """Test model initialization."""
        forest = TemporalForest()
        self.assertIsNotNone(forest.model)
        self.assertFalse(forest.is_trained)
    
    def test_training(self):
        """Test model training."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 4, 100)
        
        forest = TemporalForest(n_estimators=10)
        results = forest.train(X, y)
        
        self.assertTrue(forest.is_trained)
        self.assertIn('train_macro_f1', results)
        self.assertIn('test_macro_f1', results)

if __name__ == '__main__':
    unittest.main()

