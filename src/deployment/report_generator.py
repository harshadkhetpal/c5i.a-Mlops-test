import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

class ReportGenerator:
    """Generate automated batch reports."""
    
    def __init__(self):
        self.reports = []
    
    def generate_batch_report(self, 
                             data: pd.DataFrame,
                             batch_id: str = 'batch_id',
                             metrics: Optional[pd.DataFrame] = None,
                             anomalies: Optional[pd.DataFrame] = None,
                             phase_forecast: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive batch report."""
        
        report = {
            'batch_id': batch_id,
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'metrics': {},
            'anomalies': {},
            'recommendations': []
        }
        
        # Basic summary
        report['summary'] = {
            'total_samples': len(data),
            'start_time': str(data['timestamp_index'].min()) if 'timestamp_index' in data.columns else None,
            'end_time': str(data['timestamp_index'].max()) if 'timestamp_index' in data.columns else None,
            'duration_hours': None
        }
        
        if 'timestamp_index' in data.columns:
            start = pd.to_datetime(data['timestamp_index'].min())
            end = pd.to_datetime(data['timestamp_index'].max())
            report['summary']['duration_hours'] = (end - start).total_seconds() / 3600
        
        # Metrics
        if metrics is not None and len(metrics) > 0:
            report['metrics'] = metrics.iloc[0].to_dict()
            # Convert numpy types
            for key, value in report['metrics'].items():
                if isinstance(value, (np.integer, np.floating)):
                    report['metrics'][key] = float(value) if not np.isnan(value) else None
        
        # Anomalies
        if anomalies is not None and len(anomalies) > 0:
            report['anomalies'] = {
                'count': len(anomalies),
                'types': anomalies['anomaly_type'].value_counts().to_dict() if 'anomaly_type' in anomalies.columns else {},
                'severity_breakdown': anomalies['severity'].value_counts().to_dict() if 'severity' in anomalies.columns else {}
            }
        else:
            report['anomalies'] = {
                'count': 0,
                'types': {},
                'severity_breakdown': {}
            }
        
        # Phase forecast
        if phase_forecast:
            report['phase_forecast'] = phase_forecast
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(data, metrics, anomalies)
        
        self.reports.append(report)
        return report
    
    def _generate_recommendations(self, 
                                 data: pd.DataFrame,
                                 metrics: Optional[pd.DataFrame],
                                 anomalies: Optional[pd.DataFrame]) -> List[str]:
        """Generate data-driven recommendations."""
        recommendations = []
        
        # Check for stuck fermentation
        if anomalies is not None and len(anomalies) > 0:
            stuck_count = len(anomalies[anomalies['anomaly_type'] == 'stuck_fermentation']) if 'anomaly_type' in anomalies.columns else 0
            if stuck_count > 0:
                recommendations.append("Stuck fermentation detected. Consider checking yeast viability and temperature.")
        
        # Check pressure anomalies
        if anomalies is not None and len(anomalies) > 0:
            pressure_count = len(anomalies[anomalies['anomaly_type'].str.contains('pressure', case=False)]) if 'anomaly_type' in anomalies.columns else 0
            if pressure_count > 0:
                recommendations.append("Pressure anomalies detected. Monitor tank pressure closely and check safety valves.")
        
        # Check oxidation risks
        if anomalies is not None and len(anomalies) > 0:
            oxidation_count = len(anomalies[anomalies['anomaly_type'] == 'oxidation_risk']) if 'anomaly_type' in anomalies.columns else 0
            if oxidation_count > 0:
                recommendations.append("Oxidation risk detected. Check tank seals and DO levels.")
        
        # Check metrics
        if metrics is not None and len(metrics) > 0:
            metric = metrics.iloc[0]
            
            if 'peak_co2_ppm' in metric:
                peak_co2 = metric['peak_co2_ppm']
                if peak_co2 and peak_co2 < 1000:
                    recommendations.append("Low peak CO2 detected. Consider reviewing fermentation parameters.")
            
            if 'duration_hours' in metric:
                duration = metric['duration_hours']
                if duration and duration > 200:
                    recommendations.append("Extended fermentation duration. Review process efficiency.")
        
        if not recommendations:
            recommendations.append("Batch appears normal. Continue monitoring.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Report - {report.get('batch_id', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .anomaly {{ color: red; }}
                .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
            </style>
        </head>
        <body>
            <h1>Batch Report: {report.get('batch_id', 'Unknown')}</h1>
            <p>Generated at: {report.get('generated_at', 'Unknown')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Total Samples: {report['summary'].get('total_samples', 'N/A')}</p>
                <p>Duration: {report['summary'].get('duration_hours', 'N/A')} hours</p>
            </div>
            
            <div class="section">
                <h2>Metrics</h2>
                {self._format_metrics(report.get('metrics', {}))}
            </div>
            
            <div class="section">
                <h2>Anomalies</h2>
                <p>Total Anomalies: {report['anomalies'].get('count', 0)}</p>
                {self._format_anomalies(report.get('anomalies', {}))}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._format_recommendations(report.get('recommendations', []))}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for HTML."""
        html = "<ul>"
        for key, value in metrics.items():
            if value is not None:
                html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html
    
    def _format_anomalies(self, anomalies: Dict[str, Any]) -> str:
        """Format anomalies for HTML."""
        if anomalies.get('count', 0) == 0:
            return "<p>No anomalies detected.</p>"
        
        html = "<ul>"
        for anomaly_type, count in anomalies.get('types', {}).items():
            html += f"<li class='anomaly'>{anomaly_type}: {count}</li>"
        html += "</ul>"
        return html
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations for HTML."""
        html = ""
        for rec in recommendations:
            html += f"<div class='recommendation'>{rec}</div>"
        return html

