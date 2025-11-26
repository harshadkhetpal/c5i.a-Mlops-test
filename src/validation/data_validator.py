import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    check_func: Callable
    error_message: str

class DataValidator:
    """Data validation system."""
    
    def __init__(self):
        self.rules = []
        self.validation_results = []
    
    def add_schema_rule(self, column: str, dtype: type = None, nullable: bool = False):
        """Add schema validation rule."""
        def check_schema(data: pd.DataFrame) -> tuple:
            if column not in data.columns:
                return False, f"Column {column} missing"
            if dtype and not data[column].dtype == dtype:
                return False, f"Column {column} has wrong dtype: expected {dtype}, got {data[column].dtype}"
            if not nullable and data[column].isnull().any():
                null_count = data[column].isnull().sum()
                return False, f"Column {column} has {null_count} null values"
            return True, None
        
        self.rules.append(ValidationRule(
            name=f"schema_{column}",
            check_func=lambda df: check_schema(df),
            error_message=f"Schema validation failed for {column}"
        ))
    
    def add_range_rule(self, column: str, min_val: float = None, max_val: float = None):
        """Add range validation rule."""
        def check_range(data: pd.DataFrame) -> tuple:
            if column not in data.columns:
                return False, f"Column {column} missing"
            
            violations = []
            if min_val is not None:
                below_min = (data[column] < min_val).sum()
                if below_min > 0:
                    violations.append(f"{below_min} values below minimum {min_val}")
            
            if max_val is not None:
                above_max = (data[column] > max_val).sum()
                if above_max > 0:
                    violations.append(f"{above_max} values above maximum {max_val}")
            
            if violations:
                return False, "; ".join(violations)
            return True, None
        
        self.rules.append(ValidationRule(
            name=f"range_{column}",
            check_func=lambda df: check_range(df),
            error_message=f"Range validation failed for {column}"
        ))
    
    def add_duplicate_timestamp_rule(self, time_col: str = 'timestamp_index', 
                                     group_by: str = None):
        """Add duplicate timestamp check."""
        def check_duplicates(data: pd.DataFrame) -> tuple:
            if time_col not in data.columns:
                return True, None  # Skip if time column doesn't exist
            
            if group_by and group_by in data.columns:
                duplicates = data.groupby([group_by, time_col]).size()
                dup_count = (duplicates > 1).sum()
            else:
                duplicates = data.groupby(time_col).size()
                dup_count = (duplicates > 1).sum()
            
            if dup_count > 0:
                return False, f"{dup_count} duplicate timestamps found"
            return True, None
        
        self.rules.append(ValidationRule(
            name="duplicate_timestamps",
            check_func=lambda df: check_duplicates(df),
            error_message="Duplicate timestamps detected"
        ))
    
    def add_missing_detection_rule(self, threshold: float = 0.1):
        """Add missing value detection rule."""
        def check_missing(data: pd.DataFrame) -> tuple:
            missing_pct = data.isnull().sum() / len(data)
            high_missing = missing_pct[missing_pct > threshold]
            
            if len(high_missing) > 0:
                cols = high_missing.index.tolist()
                return False, f"Columns with >{threshold*100}% missing: {cols}"
            return True, None
        
        self.rules.append(ValidationRule(
            name="missing_detection",
            check_func=lambda df: check_missing(df),
            error_message="High percentage of missing values detected"
        ))
    
    def add_sensor_field_validation(self):
        """Add validation for all sensor fields from case study."""
        # Valve state validation
        self.add_valve_state_validation()
        
        # Agitator RPM validation
        self.add_agitator_rpm_validation()
        
        # CO2 range validation
        self.add_range_rule('co2_lpm', min_val=0, max_val=5000)
        
        # DO range validation
        self.add_range_rule('do_ppm', min_val=0, max_val=21)
        
        # Pressure range validation
        self.add_range_rule('pressure_bar', min_val=0.5, max_val=3.0)
        
        # Temperature range validation
        self.add_range_rule('temp_c', min_val=10, max_val=40)
    
    def add_outlier_detection_rule(self, column: str, method: str = 'iqr'):
        """Add outlier detection rule."""
        def check_outliers(data: pd.DataFrame) -> tuple:
            if column not in data.columns:
                return True, None
            
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((data[column] < lower) | (data[column] > upper)).sum()
            else:  # zscore
                mean = data[column].mean()
                std = data[column].std()
                if std > 0:
                    outliers = ((data[column] < mean - 3*std) | (data[column] > mean + 3*std)).sum()
                else:
                    outliers = 0
            
            if outliers > len(data) * 0.05:  # More than 5% outliers
                return False, f"{outliers} outliers detected in {column} ({outliers/len(data)*100:.1f}%)"
            return True, None
        
        self.rules.append(ValidationRule(
            name=f"outlier_{column}",
            check_func=lambda df: check_outliers(df),
            error_message=f"Excessive outliers detected in {column}"
        ))
    
    def add_valve_state_validation(self):
        """Add validation for valve_state (should be 0 or 1)."""
        def check_valve_state(data: pd.DataFrame) -> tuple:
            if 'valve_state' not in data.columns:
                return True, None
            
            invalid = ~data['valve_state'].isin([0, 1])
            invalid_count = invalid.sum()
            
            if invalid_count > 0:
                return False, f"{invalid_count} invalid valve_state values (must be 0 or 1)"
            return True, None
        
        self.rules.append(ValidationRule(
            name="valve_state_validation",
            check_func=lambda df: check_valve_state(df),
            error_message="Invalid valve_state values detected"
        ))
    
    def add_agitator_rpm_validation(self, min_rpm: float = 0, max_rpm: float = 3000):
        """Add validation for agitator_rpm range."""
        def check_agitator_rpm(data: pd.DataFrame) -> tuple:
            if 'agitator_rpm' not in data.columns:
                return True, None
            
            out_of_range = (data['agitator_rpm'] < min_rpm) | (data['agitator_rpm'] > max_rpm)
            out_of_range_count = out_of_range.sum()
            
            if out_of_range_count > 0:
                return False, f"{out_of_range_count} agitator_rpm values out of range [{min_rpm}, {max_rpm}]"
            return True, None
        
        self.rules.append(ValidationRule(
            name="agitator_rpm_validation",
            check_func=lambda df: check_agitator_rpm(df),
            error_message="Agitator RPM out of valid range"
        ))
    
    def generate_sensor_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for all sensor fields."""
        sensor_cols = ['co2_lpm', 'do_ppm', 'temp_c', 'pressure_bar', 
                      'valve_state', 'agitator_rpm']
        summary = {}
        
        for col in sensor_cols:
            if col in data.columns:
                summary[col] = {
                    'mean': float(data[col].mean()) if data[col].dtype in ['float64', 'int64'] else None,
                    'std': float(data[col].std()) if data[col].dtype in ['float64', 'int64'] else None,
                    'min': float(data[col].min()) if data[col].dtype in ['float64', 'int64'] else None,
                    'max': float(data[col].max()) if data[col].dtype in ['float64', 'int64'] else None,
                    'missing_count': int(data[col].isnull().sum())
                }
        
        return summary
    
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation rules."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'rule_results': []
        }
        
        for rule in self.rules:
            try:
                passed, message = rule.check_func(data)
                rule_result = {
                    'rule': rule.name,
                    'passed': passed,
                    'message': message
                }
                
                results['rule_results'].append(rule_result)
                
                if not passed:
                    results['passed'] = False
                    results['errors'].append({
                        'rule': rule.name,
                        'message': message or rule.error_message
                    })
            except Exception as e:
                results['passed'] = False
                results['errors'].append({
                    'rule': rule.name,
                    'message': f"Error executing rule: {str(e)}"
                })
        
        self.validation_results.append(results)
        return results
    
    def generate_report(self) -> str:
        """Generate validation report."""
        if not self.validation_results:
            return "No validation results available."
        
        report = "=== DATA VALIDATION REPORT ===\n\n"
        
        for i, result in enumerate(self.validation_results):
            report += f"Validation Run {i+1}:\n"
            report += f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n"
            report += f"Errors: {len(result['errors'])}\n"
            
            if result['errors']:
                report += "\nErrors:\n"
                for error in result['errors']:
                    report += f"  - {error['rule']}: {error['message']}\n"
            
            report += "\n"
        
        return report

