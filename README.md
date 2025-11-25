# Fermentation Gas Intelligence System

An end-to-end ML workflow for beer manufacturing fermentation monitoring, phase prediction, and anomaly detection.

## Overview

This system provides:
- **Phase Forecasting**: Predict fermentation phase 6 hours ahead
- **Anomaly Detection**: Detect stuck fermentation, oxidation risks, pressure anomalies, and abnormal CO2 activity
- **Batch Analytics**: Comprehensive batch-level summaries and metrics
- **Data-Driven Recommendations**: Automated recommendations based on batch performance

## Project Structure

```
c5i.a-Mlops-test/
├── data/                          # Dataset directory
│   └── gas_sensors_full_scale_dataset.csv
├── src/                           # Source code
│   ├── preprocessing/             # Data preprocessing modules
│   │   ├── base_preprocessor.py
│   │   ├── missing_handler.py
│   │   ├── outlier_handler.py
│   │   ├── normalizer.py
│   │   ├── resampler.py
│   │   ├── aligner.py
│   │   └── pipeline.py
│   ├── analytics/                 # Advanced analytics
│   │   ├── pandas_analytics.py
│   │   └── numpy_operations.py
│   ├── features/                  # Feature engineering
│   │   └── feature_engineering.py
│   ├── validation/                # Data validation
│   │   └── data_validator.py
│   ├── models/                    # ML models
│   │   ├── phase_predictor.py
│   │   └── changepoint_detector.py
│   ├── anomaly/                   # Anomaly detection
│   │   └── anomaly_detector.py
│   └── deployment/                # API and reports
│       ├── api.py
│       └── report_generator.py
├── notebooks/                     # Jupyter notebooks
├── models/                       # Trained models
├── reports/                      # Generated reports
├── main.py                       # Main execution script
└── requirements.txt              # Python dependencies
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Complete Pipeline

Execute the main script to run the entire workflow:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Validate data quality
- Compute advanced analytics
- Engineer features
- Train the phase prediction model
- Detect anomalies
- Generate batch reports

### 2. Use Individual Components

#### Preprocessing Pipeline

```python
from src.preprocessing import MissingHandler, OutlierHandler, PreprocessingPipeline

# Create pipeline
pipeline = PreprocessingPipeline([
    MissingHandler(method='both'),
    OutlierHandler(method='iqr', action='clip'),
    Resampler(freq='5T'),
    Normalizer(method='standard')
])

# Fit and transform
processed_data = pipeline.fit_transform(data)
```

#### Feature Engineering

```python
from src.features import FeatureEngineering

# Create all features
feature_data = FeatureEngineering.create_all_features(data)
```

#### Phase Prediction

```python
from src.models import PhasePredictor

# Load model
predictor = PhasePredictor()
predictor.load('models/phase_predictor.pkl')

# Forecast
forecast = predictor.forecast(data, hours_ahead=6)
```

#### Anomaly Detection

```python
from src.anomaly import AnomalyDetector

detector = AnomalyDetector()
anomalies = detector.detect_all(data)
```

### 3. API Deployment

Start the Flask API server:

```python
from src.deployment import create_app
from src.models import PhasePredictor
from src.anomaly import AnomalyDetector

# Load models
predictor = PhasePredictor()
predictor.load('models/phase_predictor.pkl')

detector = AnomalyDetector()

# Create app
app = create_app(predictor, detector)
app.run(host='0.0.0.0', port=5000)
```

#### API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict phase forecast
  ```json
  {
    "sensor_data": [
      {"timestamp_index": "2025-01-01 00:00:00", "co2_ppm": 1000, ...}
    ]
  }
  ```
- `POST /detect_anomalies` - Detect anomalies
- `POST /batch_summary` - Generate batch summary

## Key Features

### 1. Preprocessing Pipeline

- **MissingHandler**: Interpolation and forward fill
- **OutlierHandler**: IQR and Z-score outlier detection
- **Normalizer**: Standard, MinMax, or Robust scaling
- **Resampler**: Uniform time intervals (5 minutes)
- **Aligner**: Align to golden profiles

### 2. Advanced Analytics

- Batch-level metrics (peak CO2, time to peak, DO half-life)
- Rolling statistics (mean, std, kurtosis, CV)
- Pivot tables for batch comparison
- NumPy vectorization and broadcasting

### 3. Feature Engineering

- Polynomial features
- Interaction terms (CO2 × temperature)
- Lag features (5, 15, 60 minutes)
- Rolling statistical features
- Temporal features (hour, day, cyclical encoding)
- Phase binning

### 4. Data Validation

- Schema validation
- Range validation
- Duplicate timestamp detection
- Missing value detection
- Outlier detection rules

### 5. Modeling

- **Phase Predictor**: Gradient Boosting Machine for phase classification
- **Changepoint Detector**: Detect phase boundaries
- Evaluation using Macro-F1 score

### 6. Anomaly Detection

- Stuck fermentation detection
- Oxidation risk detection
- Pressure anomaly detection
- Abnormal CO2 activity detection
- Anomaly timeline generation

### 7. Deployment

- RESTful API for real-time predictions
- Automated batch report generation (JSON and HTML)
- Data-driven recommendations

## Data Format

The system expects time-series data with the following columns:

- `timestamp_index`: Timestamp (datetime)
- `co2_ppm`: CO2 concentration (ppm)
- `o2_pct`: Oxygen percentage
- `pressure_kpa`: Pressure (kPa)
- `process_temp_c`: Process temperature (°C)
- Additional sensor columns as available

## Outputs

After running `main.py`, you'll find:

- **models/phase_predictor.pkl**: Trained phase prediction model
- **reports/batch_*_report.json**: Batch reports in JSON format
- Console output with processing status and metrics

## Example Notebook

See `notebooks/exploration.ipynb` for detailed examples and visualizations.

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue.

