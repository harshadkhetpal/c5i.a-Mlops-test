"""
Main execution script for Fermentation Gas Intelligence System
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import (
    MissingHandler, OutlierHandler, Normalizer, 
    Resampler, Aligner, PreprocessingPipeline
)
from src.analytics import PandasAnalytics, NumPyOperations
from src.features import FeatureEngineering
from src.validation import DataValidator
from src.models import PhasePredictor, ChangepointDetector
from src.anomaly import AnomalyDetector
from src.deployment import ReportGenerator

def load_data(data_path: str = 'gas_sensors_full_scale_dataset.csv') -> pd.DataFrame:
    """Load the dataset."""
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} rows and {len(data.columns)} columns")
    return data

def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data by adding synthetic batch/tank IDs if needed."""
    data = data.copy()
    
    # Create synthetic batch_id based on date if not present
    if 'batch_id' not in data.columns:
        data['batch_id'] = pd.to_datetime(data['timestamp_index']).dt.date.astype(str)
    
    # Create synthetic strain and style for alignment
    if 'strain' not in data.columns:
        data['strain'] = 'default_strain'
    if 'style' not in data.columns:
        data['style'] = 'default_style'
    
    return data

def run_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Run preprocessing pipeline."""
    print("\n=== Running Preprocessing Pipeline ===")
    
    # Create preprocessing steps
    missing_handler = MissingHandler(method='both', limit=3)
    outlier_handler = OutlierHandler(method='iqr', action='clip')
    normalizer = Normalizer(method='standard', group_by=None)  # Global normalization
    resampler = Resampler(freq='5min', method='mean', time_col='timestamp_index')
    aligner = Aligner(golden_profiles=None)
    
    # Create pipeline
    pipeline = PreprocessingPipeline([
        missing_handler,
        outlier_handler,
        resampler,
        normalizer,
        aligner
    ])
    
    # Fit and transform
    processed_data = pipeline.fit_transform(data)
    
    # Print reports
    print("\nPreprocessing Reports:")
    for report in pipeline.get_reports():
        print(f"  - {report.get('preprocessor', 'Unknown')}: Completed")
    
    return processed_data

def run_validation(data: pd.DataFrame) -> bool:
    """Run data validation."""
    print("\n=== Running Data Validation ===")
    
    validator = DataValidator()
    
    # Add validation rules
    validator.add_schema_rule('timestamp_index', nullable=False)
    validator.add_schema_rule('co2_ppm', nullable=False)
    validator.add_range_rule('co2_ppm', min_val=0, max_val=5000)
    validator.add_range_rule('pressure_kpa', min_val=50, max_val=200)
    validator.add_duplicate_timestamp_rule('timestamp_index')
    validator.add_missing_detection_rule(threshold=0.1)
    
    # Run validation
    results = validator.validate(data)
    
    print(f"Validation Status: {'PASSED' if results['passed'] else 'FAILED'}")
    if results['errors']:
        print("Errors:")
        for error in results['errors']:
            print(f"  - {error['message']}")
    
    return results['passed']

def run_analytics(data: pd.DataFrame):
    """Run advanced analytics."""
    print("\n=== Running Advanced Analytics ===")
    
    # Compute batch metrics
    batch_metrics = PandasAnalytics.compute_batch_metrics(data)
    print(f"Computed metrics for {len(batch_metrics)} batches")
    
    # Compute rolling statistics
    data_with_rolling = PandasAnalytics.compute_rolling_statistics(data, window=12)
    print("Computed rolling statistics")
    
    # Compute correlation matrix
    corr_matrix = NumPyOperations.pearson_correlation_matrix(data)
    print(f"Computed correlation matrix: {corr_matrix.shape}")
    
    return batch_metrics, data_with_rolling

def run_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering."""
    print("\n=== Running Feature Engineering ===")
    
    data_with_features = FeatureEngineering.create_all_features(data)
    
    feature_count = len([c for c in data_with_features.columns if c not in data.columns])
    print(f"Created {feature_count} new features")
    
    return data_with_features

def train_model(data: pd.DataFrame):
    """Train phase prediction model."""
    print("\n=== Training Phase Prediction Model ===")
    
    # Prepare features
    feature_data = FeatureEngineering.create_all_features(data)
    
    # Create target (phase)
    if 'phase' not in feature_data.columns:
        # Create synthetic phases based on CO2 levels
        feature_data['phase'] = pd.cut(
            feature_data['co2_ppm'],
            bins=[0, 500, 1000, 1500, float('inf')],
            labels=['lag', 'exponential', 'stationary', 'decline']
        )
    
    # Encode phases
    phase_map = {'lag': 0, 'exponential': 1, 'stationary': 2, 'decline': 3}
    feature_data['phase_encoded'] = feature_data['phase'].map(phase_map).fillna(0)
    
    # Prepare training data
    exclude_cols = ['timestamp_index', 'batch_id', 'phase', 'phase_encoded', 'strain', 'style']
    feature_cols = [c for c in feature_data.select_dtypes(include=[np.number]).columns 
                   if c not in exclude_cols]
    
    X = feature_data[feature_cols].fillna(0).values
    y = feature_data['phase_encoded'].values
    
    # Train model
    predictor = PhasePredictor(n_estimators=50, learning_rate=0.1, max_depth=3)
    # Set feature columns before training so they're saved
    predictor.feature_columns = feature_cols
    results = predictor.train(X, y, test_size=0.2)
    
    print(f"Train Macro-F1: {results['train_macro_f1']:.4f}")
    print(f"Test Macro-F1: {results['test_macro_f1']:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    predictor.save('models/phase_predictor.pkl')
    print("Model saved to models/phase_predictor.pkl")
    
    return predictor, feature_data

def run_anomaly_detection(data: pd.DataFrame):
    """Run anomaly detection."""
    print("\n=== Running Anomaly Detection ===")
    
    detector = AnomalyDetector()
    anomalies = detector.detect_all(data)
    
    print(f"Detected {len(anomalies)} anomalies")
    if len(anomalies) > 0:
        print("Anomaly types:")
        print(anomalies['anomaly_type'].value_counts())
    
    return detector, anomalies

def generate_reports(data: pd.DataFrame, batch_metrics: pd.DataFrame, 
                    anomalies: pd.DataFrame, predictor: PhasePredictor):
    """Generate batch reports."""
    print("\n=== Generating Reports ===")
    
    report_gen = ReportGenerator()
    
    # Generate report for first batch
    if 'batch_id' in data.columns:
        batch_ids = data['batch_id'].unique()[:3]  # First 3 batches
        
        for batch_id in batch_ids:
            batch_data = data[data['batch_id'] == batch_id]
            batch_metric = batch_metrics[batch_metrics['batch_id'] == batch_id] if len(batch_metrics) > 0 else None
            batch_anomalies = anomalies[anomalies.get('batch_id', pd.Series()) == batch_id] if len(anomalies) > 0 else None
            
            # Forecast
            try:
                forecast = predictor.forecast(batch_data)
            except:
                forecast = None
            
            report = report_gen.generate_batch_report(
                batch_data, batch_id, batch_metric, batch_anomalies, forecast
            )
            
            # Save report
            os.makedirs('reports', exist_ok=True)
            report_gen.save_report(report, f'reports/batch_{batch_id}_report.json')
            print(f"Generated report for batch {batch_id}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("FERMENTATION GAS INTELLIGENCE SYSTEM")
    print("=" * 60)
    
    # Load data
    data = load_data()
    data = prepare_data(data)
    
    # Run preprocessing
    processed_data = run_preprocessing(data)
    
    # Run validation
    validation_passed = run_validation(processed_data)
    if not validation_passed:
        print("Warning: Data validation failed, but continuing...")
    
    # Run analytics
    batch_metrics, data_with_rolling = run_analytics(processed_data)
    
    # Run feature engineering
    feature_data = run_feature_engineering(data_with_rolling)
    
    # Train model
    predictor, training_data = train_model(feature_data)
    
    # Run anomaly detection
    detector, anomalies = run_anomaly_detection(processed_data)
    
    # Generate reports
    generate_reports(processed_data, batch_metrics, anomalies, predictor)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - Model: models/phase_predictor.pkl")
    print("  - Reports: reports/batch_*_report.json")
    print("\nTo start the API server, run:")
    print("  python -m src.deployment.api")

if __name__ == '__main__':
    main()

