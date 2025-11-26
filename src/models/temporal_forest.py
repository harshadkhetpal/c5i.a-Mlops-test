"""
Temporal Forest - Random Forest-based time-series model for phase prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from typing import Dict, List, Optional
import joblib

class TemporalForest:
    """Temporal Random Forest for phase prediction with time-series awareness."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_columns = None
        self.is_trained = False
        self.classes_ = None
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Prepare features with temporal awareness."""
        if feature_cols is None:
            if self.feature_columns is not None:
                feature_cols = self.feature_columns
            else:
                exclude_cols = ['timestamp', 'timestamp_index', 'batch_id', 'tank_id', 
                               'phase', 'phase_encoded', 'strain', 'style']
                feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns 
                               if c not in exclude_cols]
                self.feature_columns = feature_cols
        
        available_cols = [c for c in feature_cols if c in data.columns]
        
        if len(available_cols) < len(feature_cols):
            result_df = pd.DataFrame(index=data.index)
            for col in feature_cols:
                if col in data.columns:
                    result_df[col] = data[col]
                else:
                    result_df[col] = 0
            return result_df[feature_cols].fillna(0).values
        
        if len(available_cols) == 0:
            raise ValueError("No valid feature columns found")
        
        return data[available_cols].fillna(0).values
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2) -> Dict[str, float]:
        """Train the temporal forest model."""
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        unique_classes, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        use_stratify = min_class_count >= 2
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if use_stratify else None
        )
        
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        # Set feature columns if not set
        if self.feature_columns is None:
            self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_f1 = f1_score(y_train, y_pred_train, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        
        self.is_trained = True
        
        # Feature importance
        feature_importance = {}
        if self.feature_columns and len(self.feature_columns) == len(self.model.feature_importances_):
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_.tolist()
            ))
        
        return {
            'train_macro_f1': train_f1,
            'test_macro_f1': test_f1,
            'classification_report': classification_report(y_test, y_pred_test, zero_division=0),
            'feature_importance': feature_importance
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict phase."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict phase probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X)
    
    def forecast(self, data: pd.DataFrame, hours_ahead: int = 6) -> Dict[str, any]:
        """Forecast phase 6 hours ahead."""
        X = self.prepare_features(data)
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        last_pred = predictions[-1]
        last_proba = probabilities[-1]
        
        # Convert numpy types
        if isinstance(last_pred, (np.integer, np.int64)):
            last_pred = int(last_pred)
        elif isinstance(last_pred, np.ndarray):
            last_pred = int(last_pred.item()) if last_pred.size == 1 else int(last_pred[-1])
        
        phase_map = {0: 'lag', 1: 'exponential', 2: 'stationary', 3: 'decline'}
        predicted_phase_name = phase_map.get(int(last_pred), str(last_pred))
        
        proba_dict = {}
        for i in range(len(self.classes_)):
            class_name = str(self.classes_[i]) if isinstance(self.classes_[i], (np.integer, np.int64)) else self.classes_[i]
            proba_dict[class_name] = float(last_proba[i])
        
        return {
            'predicted_phase': predicted_phase_name,
            'phase_probabilities': proba_dict,
            'confidence': float(max(last_proba))
        }
    
    def save(self, filepath: str):
        """Save the model."""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'classes_': self.classes_
        }, filepath)
    
    def load(self, filepath: str):
        """Load the model."""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.feature_columns = loaded['feature_columns']
        self.is_trained = loaded['is_trained']
        self.classes_ = loaded['classes_']

