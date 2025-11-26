"""
Schema Mapper - Maps between case study schema and actual dataset schema
"""
import pandas as pd
from typing import Dict, Optional

class SchemaMapper:
    """Map between case study specification and actual dataset columns."""
    
    # Case study column names -> Actual dataset column names
    COLUMN_MAPPING = {
        # Telemetry columns
        'co2_lpm': 'co2_ppm',  # CO2 liters per minute -> CO2 ppm (approximate mapping)
        'do_ppm': 'o2_pct',    # Dissolved oxygen ppm -> O2 percentage
        'temp_c': 'process_temp_c',  # Temperature
        'pressure_bar': 'pressure_kpa',  # Pressure bar -> kPa (1 bar = 100 kPa)
        'valve_state': None,  # Not in dataset - will create synthetic
        'agitator_rpm': None,  # Not in dataset - will create synthetic
        
        # Metadata columns
        'pitch_time': None,  # Not in dataset - will create synthetic
        'OG': None,  # Original Gravity - not in dataset
        'target_attenuation': None,  # Not in dataset
        
        # Keep existing mappings
        'timestamp': 'timestamp_index',
        'tank_id': 'batch_id',  # Use batch_id as tank_id
        'batch_id': 'batch_id',
        'strain': 'strain',
        'style': 'style',
    }
    
    # Unit conversions
    PRESSURE_CONVERSION = 100.0  # bar to kPa (1 bar = 100 kPa)
    CO2_CONVERSION = 1.0  # Approximate: lpm to ppm (needs calibration)
    
    @staticmethod
    def map_to_case_study_schema(data: pd.DataFrame) -> pd.DataFrame:
        """Map actual dataset columns to case study schema."""
        data = data.copy()
        
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in SchemaMapper.COLUMN_MAPPING.items() 
                          if v is not None and v in data.columns}
        
        # Rename columns
        data = data.rename(columns=reverse_mapping)
        
        # Convert units
        if 'pressure_bar' in data.columns:
            # Convert from kPa to bar
            data['pressure_bar'] = data['pressure_bar'] / SchemaMapper.PRESSURE_CONVERSION
        
        # Create synthetic columns if missing
        if 'valve_state' not in data.columns:
            # Synthetic valve state based on pressure (0=closed, 1=open)
            if 'pressure_bar' in data.columns:
                data['valve_state'] = (data['pressure_bar'] > 1.0).astype(int)
            else:
                data['valve_state'] = 1
        
        if 'agitator_rpm' not in data.columns:
            # Synthetic agitator RPM based on process activity
            if 'process_temp_c' in data.columns:
                # Higher temp = more agitation
                data['agitator_rpm'] = (data['process_temp_c'] * 20).clip(0, 2000)
            else:
                data['agitator_rpm'] = 1000
        
        if 'pitch_time' not in data.columns:
            # Use first timestamp as pitch time
            if 'timestamp' in data.columns:
                data['pitch_time'] = pd.to_datetime(data['timestamp']).min()
            else:
                data['pitch_time'] = pd.Timestamp.now()
        
        if 'OG' not in data.columns:
            # Synthetic OG based on initial CO2 (higher OG = more fermentable sugars)
            if 'co2_lpm' in data.columns:
                initial_co2 = data['co2_lpm'].iloc[0] if len(data) > 0 else 0
                data['OG'] = 1.040 + (initial_co2 / 10000)  # Approximate
            else:
                data['OG'] = 1.050
        
        if 'target_attenuation' not in data.columns:
            # Standard target attenuation
            data['target_attenuation'] = 75.0
        
        return data
    
    @staticmethod
    def map_from_case_study_schema(data: pd.DataFrame) -> pd.DataFrame:
        """Map case study schema back to actual dataset columns."""
        data = data.copy()
        
        # Apply forward mapping
        mapping = {k: v for k, v in SchemaMapper.COLUMN_MAPPING.items() 
                  if v is not None and k in data.columns}
        
        data = data.rename(columns=mapping)
        
        # Convert units back
        if 'pressure_kpa' in data.columns and 'pressure_bar' in data.columns:
            # Already converted, remove bar column
            if 'pressure_bar' in data.columns:
                del data['pressure_bar']
        
        return data
    
    @staticmethod
    def get_case_study_columns() -> Dict[str, str]:
        """Get expected case study column names."""
        return {
            'telemetry': ['timestamp', 'tank_id', 'batch_id', 'strain', 'style', 
                         'co2_lpm', 'do_ppm', 'temp_c', 'pressure_bar', 
                         'valve_state', 'agitator_rpm'],
            'metadata': ['batch_id', 'pitch_time', 'OG', 'target_attenuation']
        }

