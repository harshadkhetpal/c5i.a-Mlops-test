from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from src.features import FeatureEngineering
from src.preprocessing import SchemaMapper

# Request/Response models
class SensorDataPoint(BaseModel):
    timestamp: str  # Case study schema
    co2_lpm: Optional[float] = None  # Case study schema
    do_ppm: Optional[float] = None  # Case study schema
    temp_c: Optional[float] = None  # Case study schema
    pressure_bar: Optional[float] = None  # Case study schema
    valve_state: Optional[int] = None  # Case study schema (0 or 1)
    agitator_rpm: Optional[float] = None  # Case study schema
    # Legacy support - will be mapped
    timestamp_index: Optional[str] = None
    co2_ppm: Optional[float] = None
    o2_pct: Optional[float] = None
    pressure_kpa: Optional[float] = None
    process_temp_c: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    # Metadata fields
    batch_id: Optional[str] = None
    tank_id: Optional[str] = None
    strain: Optional[str] = None
    style: Optional[str] = None
    pitch_time: Optional[str] = None
    OG: Optional[float] = None
    target_attenuation: Optional[float] = None
    class Config:
        extra = "allow"  # Allow additional fields

class PredictionRequest(BaseModel):
    sensor_data: List[SensorDataPoint]

class PredictionResponse(BaseModel):
    success: bool
    forecast: Dict[str, Any]

class AnomalyResponse(BaseModel):
    success: bool
    anomalies: List[Dict[str, Any]]
    count: int

class BatchSummaryResponse(BaseModel):
    success: bool
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str

def create_app(phase_predictor=None, anomaly_detector=None):
    """Create FastAPI application."""
    app = FastAPI(
        title="Fermentation Gas Intelligence System API",
        description="ML-powered API for fermentation phase prediction and anomaly detection",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "Fermentation Gas Intelligence System",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Predict phase forecast 6 hours ahead."""
        try:
            # Convert to DataFrame
            sensor_data = pd.DataFrame([item.dict() for item in request.sensor_data])
            
            if phase_predictor is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            # Map to case study schema if needed
            if 'timestamp_index' in sensor_data.columns and 'timestamp' not in sensor_data.columns:
                sensor_data['timestamp'] = sensor_data['timestamp_index']
            if 'co2_ppm' in sensor_data.columns and 'co2_lpm' not in sensor_data.columns:
                sensor_data['co2_lpm'] = sensor_data['co2_ppm']
            if 'o2_pct' in sensor_data.columns and 'do_ppm' not in sensor_data.columns:
                sensor_data['do_ppm'] = sensor_data['o2_pct']
            if 'pressure_kpa' in sensor_data.columns and 'pressure_bar' not in sensor_data.columns:
                sensor_data['pressure_bar'] = sensor_data['pressure_kpa'] / 100.0
            if 'process_temp_c' in sensor_data.columns and 'temp_c' not in sensor_data.columns:
                sensor_data['temp_c'] = sensor_data['process_temp_c']
            
            # Map to case study schema
            sensor_data = SchemaMapper.map_to_case_study_schema(sensor_data)
            
            # Apply feature engineering before prediction
            sensor_data = FeatureEngineering.create_all_features(sensor_data)
            
            # Use the model's prepare_features method which handles feature selection
            # Prepare features and predict
            forecast = phase_predictor.forecast(sensor_data, hours_ahead=6)
            
            return {
                "success": True,
                "forecast": forecast
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/detect_anomalies", response_model=AnomalyResponse)
    async def detect_anomalies(request: PredictionRequest):
        """Detect anomalies in sensor data."""
        try:
            # Convert to DataFrame
            sensor_data = pd.DataFrame([item.dict() for item in request.sensor_data])
            
            if anomaly_detector is None:
                raise HTTPException(status_code=500, detail="Anomaly detector not loaded")
            
            # Map to case study schema if needed
            if 'timestamp_index' in sensor_data.columns and 'timestamp' not in sensor_data.columns:
                sensor_data['timestamp'] = sensor_data['timestamp_index']
            if 'co2_ppm' in sensor_data.columns and 'co2_lpm' not in sensor_data.columns:
                sensor_data['co2_lpm'] = sensor_data['co2_ppm']
            if 'o2_pct' in sensor_data.columns and 'do_ppm' not in sensor_data.columns:
                sensor_data['do_ppm'] = sensor_data['o2_pct']
            if 'pressure_kpa' in sensor_data.columns and 'pressure_bar' not in sensor_data.columns:
                sensor_data['pressure_bar'] = sensor_data['pressure_kpa'] / 100.0
            if 'process_temp_c' in sensor_data.columns and 'temp_c' not in sensor_data.columns:
                sensor_data['temp_c'] = sensor_data['process_temp_c']
            
            # Map to case study schema
            sensor_data = SchemaMapper.map_to_case_study_schema(sensor_data)
            
            # Detect anomalies with case study column names
            anomalies = anomaly_detector.detect_all(
                sensor_data,
                co2_col='co2_lpm',
                do_col='do_ppm',
                pressure_col='pressure_bar',
                time_col='timestamp'
            )
            
            # Convert to JSON-serializable format
            if len(anomalies) > 0:
                anomalies_json = anomalies.to_dict('records')
                # Convert numpy types to Python types
                for record in anomalies_json:
                    for key, value in record.items():
                        if isinstance(value, (np.integer, np.floating)):
                            record[key] = float(value)
                        elif isinstance(value, pd.Timestamp):
                            record[key] = value.isoformat()
                        elif pd.isna(value):
                            record[key] = None
            else:
                anomalies_json = []
            
            return {
                "success": True,
                "anomalies": anomalies_json,
                "count": len(anomalies)
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_summary", response_model=BatchSummaryResponse)
    async def batch_summary(request: PredictionRequest):
        """Generate batch summary with metrics."""
        try:
            # Convert to DataFrame
            sensor_data = pd.DataFrame([item.dict() for item in request.sensor_data])
            
            # Compute batch metrics
            from src.analytics.pandas_analytics import PandasAnalytics
            
            metrics = PandasAnalytics.compute_batch_metrics(sensor_data)
            
            if len(metrics) > 0:
                summary = metrics.iloc[0].to_dict()
                # Convert numpy types
                for key, value in summary.items():
                    if isinstance(value, (np.integer, np.floating)):
                        summary[key] = float(value) if not np.isnan(value) else None
                    elif pd.isna(value):
                        summary[key] = None
            else:
                summary = {}
            
            return {
                "success": True,
                "summary": summary
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app
