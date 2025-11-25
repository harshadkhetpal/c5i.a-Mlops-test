from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from typing import Dict, Any
import json

def create_app(phase_predictor=None, anomaly_detector=None):
    """Create Flask API application."""
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy', 'service': 'Fermentation Gas Intelligence System'})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict phase forecast."""
        try:
            data = request.get_json()
            
            if not data or 'sensor_data' not in data:
                return jsonify({'error': 'Missing sensor_data'}), 400
            
            # Convert to DataFrame
            sensor_data = pd.DataFrame(data['sensor_data'])
            
            if phase_predictor is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Prepare features and predict
            X = phase_predictor.prepare_features(sensor_data)
            forecast = phase_predictor.forecast(sensor_data, hours_ahead=6)
            
            return jsonify({
                'success': True,
                'forecast': forecast
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/detect_anomalies', methods=['POST'])
    def detect_anomalies():
        """Detect anomalies in sensor data."""
        try:
            data = request.get_json()
            
            if not data or 'sensor_data' not in data:
                return jsonify({'error': 'Missing sensor_data'}), 400
            
            # Convert to DataFrame
            sensor_data = pd.DataFrame(data['sensor_data'])
            
            if anomaly_detector is None:
                return jsonify({'error': 'Anomaly detector not loaded'}), 500
            
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
            else:
                anomalies_json = []
            
            return jsonify({
                'success': True,
                'anomalies': anomalies_json,
                'count': len(anomalies)
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/batch_summary', methods=['POST'])
    def batch_summary():
        """Generate batch summary."""
        try:
            data = request.get_json()
            
            if not data or 'sensor_data' not in data:
                return jsonify({'error': 'Missing sensor_data'}), 400
            
            # Convert to DataFrame
            sensor_data = pd.DataFrame(data['sensor_data'])
            
            # Compute batch metrics
            from src.analytics.pandas_analytics import PandasAnalytics
            
            metrics = PandasAnalytics.compute_batch_metrics(sensor_data)
            
            if len(metrics) > 0:
                summary = metrics.iloc[0].to_dict()
                # Convert numpy types
                for key, value in summary.items():
                    if isinstance(value, (np.integer, np.floating)):
                        summary[key] = float(value) if not np.isnan(value) else None
            else:
                summary = {}
            
            return jsonify({
                'success': True,
                'summary': summary
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

