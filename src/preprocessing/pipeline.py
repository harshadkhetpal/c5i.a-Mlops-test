from typing import List, Dict, Any
import pandas as pd
from .base_preprocessor import BasePreprocessor

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline."""
    
    def __init__(self, steps: List[BasePreprocessor]):
        self.steps = steps
        self.reports_ = []
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'PreprocessingPipeline':
        """Fit all preprocessing steps."""
        current_data = data.copy()
        
        for step in self.steps:
            current_data = step.fit(current_data, **kwargs).transform(current_data, **kwargs)
            self.reports_.append(step.report())
        
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data through all steps."""
        current_data = data.copy()
        
        for step in self.steps:
            current_data = step.transform(current_data, **kwargs)
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def get_reports(self) -> List[Dict[str, Any]]:
        """Get all preprocessing reports."""
        return self.reports_

