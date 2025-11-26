"""
FastAPI Server Startup Script
Run this to start the FastAPI server
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.deployment import create_app
from src.models import PhasePredictor
from src.anomaly import AnomalyDetector
import uvicorn

def main():
    """Start the FastAPI server."""
    print("=" * 60)
    print("FERMENTATION GAS INTELLIGENCE SYSTEM - FASTAPI SERVER")
    print("=" * 60)
    
    # Load models
    print("\nLoading models...")
    predictor = PhasePredictor()
    
    if os.path.exists('models/phase_predictor.pkl'):
        predictor.load('models/phase_predictor.pkl')
        print("✓ Phase predictor model loaded")
    else:
        print("⚠ Warning: Model not found. Train the model first using main.py")
    
    detector = AnomalyDetector()
    print("✓ Anomaly detector initialized")
    
    # Create FastAPI app
    app = create_app(predictor, detector)
    
    print("\n" + "=" * 60)
    print("API Server Starting...")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  POST /predict          - Phase forecast (6 hours ahead)")
    print("  POST /detect_anomalies - Detect anomalies")
    print("  POST /batch_summary    - Generate batch summary")
    print("\n📚 Interactive API Documentation:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print("\nServer running on: http://localhost:8000")
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == '__main__':
    main()

