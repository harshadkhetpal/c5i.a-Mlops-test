import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.interpolate import interp1d

class NumPyOperations:
    """NumPy vectorization and broadcasting operations."""
    
    @staticmethod
    def cosine_similarity(live_curve: np.ndarray, golden_curve: np.ndarray) -> float:
        """Compute cosine similarity between live and golden curves."""
        # Normalize vectors
        live_norm = live_curve / (np.linalg.norm(live_curve) + 1e-10)
        golden_norm = golden_curve / (np.linalg.norm(golden_curve) + 1e-10)
        
        return np.dot(live_norm, golden_norm)
    
    @staticmethod
    def euclidean_distance_matrix(data: pd.DataFrame, 
                                   columns: List[str] = None) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix using broadcasting."""
        if columns is None:
            columns = ['co2_ppm', 'o2_pct', 'process_temp_c', 'pressure_kpa']
            columns = [c for c in columns if c in data.columns]
        
        # Extract feature matrix
        X = data[columns].values
        
        # Compute distance matrix using broadcasting
        # X shape: (n_samples, n_features)
        # We want (n_samples, n_samples) distance matrix
        X_expanded = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
        X_broadcast = X[np.newaxis, :, :]  # (1, n_samples, n_features)
        
        distances = np.sqrt(np.sum((X_expanded - X_broadcast) ** 2, axis=2))
        
        return distances
    
    @staticmethod
    def pearson_correlation_matrix(data: pd.DataFrame, 
                                    columns: List[str] = None) -> np.ndarray:
        """Compute Pearson correlation matrix."""
        if columns is None:
            columns = ['co2_ppm', 'o2_pct', 'process_temp_c']
            columns = [c for c in columns if c in data.columns]
        
        # Extract feature matrix
        X = data[columns].values
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        return corr_matrix
    
    @staticmethod
    def compare_to_golden(live_data: pd.DataFrame, golden_profile: pd.DataFrame,
                          target_col: str = 'co2_ppm') -> float:
        """Compare live batch to golden profile."""
        if target_col not in live_data.columns or target_col not in golden_profile.columns:
            return 0.0
        
        live_curve = live_data[target_col].values
        golden_curve = golden_profile[target_col].values
        
        # Interpolate to same length if needed
        if len(live_curve) != len(golden_curve):
            f = interp1d(np.arange(len(golden_curve)), golden_curve, 
                        kind='linear', fill_value='extrapolate')
            golden_curve = f(np.linspace(0, len(golden_curve)-1, len(live_curve)))
        
        return NumPyOperations.cosine_similarity(live_curve, golden_curve)
    
    @staticmethod
    def compute_tank_distances(data: pd.DataFrame, 
                               tank_col: str = 'batch_id',
                               feature_cols: List[str] = None) -> pd.DataFrame:
        """Compute distances between tanks/batches."""
        if feature_cols is None:
            feature_cols = ['co2_ppm', 'o2_pct', 'process_temp_c', 'pressure_kpa']
            feature_cols = [c for c in feature_cols if c in data.columns]
        
        if tank_col not in data.columns:
            return pd.DataFrame()
        
        # Group by tank/batch and compute mean features
        tank_features = data.groupby(tank_col)[feature_cols].mean()
        
        # Compute distance matrix
        distances = NumPyOperations.euclidean_distance_matrix(tank_features, feature_cols)
        
        # Create distance DataFrame
        distance_df = pd.DataFrame(
            distances,
            index=tank_features.index,
            columns=tank_features.index
        )
        
        return distance_df

