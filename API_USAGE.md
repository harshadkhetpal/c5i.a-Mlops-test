# FastAPI Usage Guide

## Starting the Server

```bash
python run_api.py
```

The server will start on `http://localhost:8000`

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Fermentation Gas Intelligence System",
  "timestamp": "2025-11-25T20:30:00"
}
```

### 2. Phase Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [
      {
        "timestamp_index": "2025-01-01 00:00:00",
        "co2_ppm": 1000,
        "o2_pct": 19.5,
        "pressure_kpa": 101.0,
        "process_temp_c": 30.0
      }
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "forecast": {
    "predicted_phase": "stationary",
    "phase_probabilities": {
      "lag": 0.1,
      "exponential": 0.2,
      "stationary": 0.6,
      "decline": 0.1
    },
    "confidence": 0.6
  }
}
```

### 3. Anomaly Detection

```bash
curl -X POST http://localhost:8000/detect_anomalies \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [
      {
        "timestamp_index": "2025-01-01 00:00:00",
        "co2_ppm": 1000,
        "o2_pct": 19.5,
        "pressure_kpa": 101.0,
        "process_temp_c": 30.0
      }
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "anomalies": [
    {
      "timestamp_index": "2025-01-01 00:00:00",
      "anomaly_type": "stuck_fermentation",
      "severity": "high"
    }
  ],
  "count": 1
}
```

### 4. Batch Summary

```bash
curl -X POST http://localhost:8000/batch_summary \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [
      {
        "timestamp_index": "2025-01-01 00:00:00",
        "co2_ppm": 1000,
        "o2_pct": 19.5,
        "pressure_kpa": 101.0
      }
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "batch_id": "2025-01-01",
    "peak_co2_ppm": 1500.0,
    "time_to_peak_minutes": 380.0,
    "duration_hours": 23.92
  }
}
```

## Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict phase
data = {
    "sensor_data": [
        {
            "timestamp_index": "2025-01-01 00:00:00",
            "co2_ppm": 1000,
            "o2_pct": 19.5,
            "pressure_kpa": 101.0,
            "process_temp_c": 30.0
        }
    ]
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Benefits of FastAPI

1. **Automatic Documentation**: Interactive Swagger UI and ReDoc
2. **Type Validation**: Automatic request/response validation with Pydantic
3. **Better Performance**: Async support and high performance
4. **Easy Testing**: Built-in test client
5. **Modern Python**: Uses Python type hints

