from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    def __init__(self):
        self.is_fitted = False
        self.params_ = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BasePreprocessor':
        """Fit the preprocessor on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    @abstractmethod
    def report(self) -> Dict[str, Any]:
        """Generate a report of preprocessing operations."""
        pass

