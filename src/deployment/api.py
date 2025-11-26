from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Request/Response models
class SensorDataPoint(BaseModel):
    timestamp_index: str
    co2_ppm: Optional[float] = None
    o2_pct: Optional[float] = None
    pressure_kpa: Optional[float] = None
    process_temp_c: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    # Add other fields as needed
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
            
            # Apply feature engineering before prediction
            from src.features import FeatureEngineering
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
            
            # Detect anomalies
            anomalies = anomaly_detector.detect_all(sensor_data)
            
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
