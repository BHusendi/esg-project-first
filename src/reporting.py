"""
Reporting Module

Generates comprehensive reports with descriptive analysis and feature recommendations.

Reference:
- ESG Analysis Best Practices
- Financial Report Writing Standards
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from . import config


class ReportGenerator:
    """
    Generate comprehensive analysis reports with recommendations.
    
    Parameters
    ----------
    forecaster : ARIMAForecaster
        Trained ARIMA forecaster
    original_df : pd.DataFrame
        Original dataset
    processed_df : pd.DataFrame
        Processed dataset
    diagnostics : ModelDiagnostics
        Model diagnostics results
    """
    def __init__(self, forecaster, original_df, processed_df=None, diagnostics=None):
        self.forecaster = forecaster
        self.original_df = original_df
        
        # FIX: Handle DataFrame dengan benar (tidak pakai 'or')
        if processed_df is not None:
            self.processed_df = processed_df
        else:
            self.processed_df = original_df
        
        self.diagnostics = diagnostics
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def generate_full_report(self, output_dir='outputs/reports/', save_files=True):
        """
        Generate complete report with all sections.
        
        Parameters
        ----------
        output_dir : str
            Directory to save report files
        save_files : bool
            Whether to save files to disk
        
        Returns
        -------
        dict
            Complete report dictionary
        """
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*70}")
        
        if save_files:
            os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': self.timestamp,
            'data_summary': self._generate_data_summary(),
            'statistical_analysis': self._generate_statistical_analysis(),
            'model_information': self._generate_model_information(),
            'diagnostics_summary': self._generate_diagnostics_summary(),
            'recommendations': self._generate_recommendations(),
        }
        
        if save_files:
            self._save_reports(report, output_dir)
        
        return report
    
    def _generate_data_summary(self):
        """Generate data quality and summary statistics."""
        print("\nGenerating Data Summary...")
        
        summary = {}
        
        for col in self.processed_df.columns:
            if col != 'Date':
                original_missing = self.original_df[col].isnull().sum()
                processed_missing = self.processed_df[col].isnull().sum()
                
                summary[col] = {
                    'n_observations': len(self.processed_df),
                    'original_missing': original_missing,
                    'processed_missing': processed_missing,
                    'fill_rate': ((original_missing - processed_missing) / original_missing * 100) 
                                 if original_missing > 0 else 0,
                    'mean': self.processed_df[col].mean(),
                    'std': self.processed_df[col].std(),
                    'min': self.processed_df[col].min(),
                    'max': self.processed_df[col].max(),
                    'median': self.processed_df[col].median(),
                    'q1': self.processed_df[col].quantile(0.25),
                    'q3': self.processed_df[col].quantile(0.75),
                    'skewness': self.processed_df[col].skew(),
                    'kurtosis': self.processed_df[col].kurtosis(),
                }
        
        return summary
    
    def _generate_statistical_analysis(self):
        """Generate statistical insights."""
        print("Generating Statistical Analysis...")
        
        analysis = {
            'overall_trend': self._analyze_trend(),
            'volatility': self._analyze_volatility(),
            'distribution': self._analyze_distribution(),
        }
        
        return analysis
    
    def _analyze_trend(self):
        """Analyze trend in data."""
        series = self.processed_df.iloc[:, -1]  # Use last numeric column
        
        # Simple trend: compare first third with last third
        first_third_mean = series.iloc[:len(series)//3].mean()
        last_third_mean = series.iloc[-len(series)//3:].mean()
        
        if last_third_mean > first_third_mean * 1.05:
            trend = "Upward"
        elif last_third_mean < first_third_mean * 0.95:
            trend = "Downward"
        else:
            trend = "Stable"
        
        change_pct = ((last_third_mean - first_third_mean) / first_third_mean) * 100
        
        return {
            'direction': trend,
            'change_percent': change_pct,
            'first_period_mean': first_third_mean,
            'last_period_mean': last_third_mean,
        }
    
    def _analyze_volatility(self):
        """Analyze volatility (standard deviation)."""
        series = self.processed_df.iloc[:, -1]
        
        rolling_std = series.rolling(window=6).std().mean()
        cv = (series.std() / series.mean()) * 100  # Coefficient of variation
        
        if cv < 5:
            volatility_level = "Low"
        elif cv < 15:
            volatility_level = "Moderate"
        else:
            volatility_level = "High"
        
        return {
            'rolling_std': rolling_std,
            'coefficient_of_variation': cv,
            'level': volatility_level,
        }
    
    def _analyze_distribution(self):
        """Analyze data distribution."""
        series = self.processed_df.iloc[:, -1]
        
        skewness = series.skew()
        
        if abs(skewness) < 0.5:
            distribution = "Approximately Symmetric"
        elif skewness > 0.5:
            distribution = "Right-skewed"
        else:
            distribution = "Left-skewed"
        
        return {
            'distribution_shape': distribution,
            'skewness_value': skewness,
        }
    
    def _generate_model_information(self):
        """Generate model parameters and metrics."""
        print("Generating Model Information...")
        
        model_info = {}
        
        if self.forecaster and self.forecaster.results:
            model_info = {
                'arima_order': self.forecaster.order,
                'seasonal_order': self.forecaster.seasonal_order,
                'aic': self.forecaster.results.aic,
                'bic': self.forecaster.results.bic,
                'llf': self.forecaster.results.llf,
                'nobs': self.forecaster.results.nobs,
                'model_summary': str(self.forecaster.results.summary()),
            }
        
        return model_info
    
    def _generate_diagnostics_summary(self):
        """Generate diagnostics summary."""
        print("Generating Diagnostics Summary...")
        
        if self.diagnostics:
            return self.diagnostics.diagnostics
        return {}
    
    def _generate_recommendations(self):
        """Generate feature recommendations."""
        print("Generating Recommendations...")
        
        recommendations = {
            'strengths': [],
            'improvements': [],
            'enhancements': [],
            'warnings': [],
        }
        
        # Analyze model fit
        if self.forecaster and self.forecaster.results:
            aic = self.forecaster.results.aic
            bic = self.forecaster.results.bic
            
            # Check data quality
            missing_rate = (self.original_df.isnull().sum().sum() / 
                           (len(self.original_df) * len(self.original_df.columns)))
            
            # Strengths
            if missing_rate < 0.05:
                recommendations['strengths'].append(
                    "✓ Excellent data quality (< 5% missing)"
                )
            elif missing_rate < 0.10:
                recommendations['strengths'].append(
                    "✓ Good data quality (< 10% missing)"
                )
            
            recommendations['strengths'].append(
                "✓ Model successfully fitted with ARIMA framework"
            )
            
            # Improvements
            recommendations['improvements'].append(
                "• Monitor forecasts against actual values quarterly"
            )
            recommendations['improvements'].append(
                "• Consider SARIMA for better seasonal capture"
            )
            if missing_rate > 0.05:
                recommendations['improvements'].append(
                    "• Improve data collection frequency to reduce missing values"
                )
            
            # Enhancements
            recommendations['enhancements'].append(
                "• Integrate external variables (economic indicators, policy changes)"
            )
            recommendations['enhancements'].append(
                "• Implement ensemble methods combining ARIMA with ML models"
            )
            recommendations['enhancements'].append(
                "• Develop scenario analysis (best/worst/base case)"
            )
            recommendations['enhancements'].append(
                "• Create early warning system for anomalies"
            )
            
            # Warnings
            recommendations['warnings'].append(
                f"⚠ Limited historical data ({len(self.original_df)} months)"
            )
            recommendations['warnings'].append(
                "⚠ Forecasts assume historical patterns continue"
            )
            recommendations['warnings'].append(
                "⚠ External shocks may invalidate predictions"
            )
            recommendations['warnings'].append(
                "⚠ Recommended: Update model quarterly with new data"
            )
        
        return recommendations
    
    def _save_reports(self, report, output_dir):
        """Save report files to disk."""
        print(f"\nSaving reports to: {output_dir}")
        
        # Save data summary
        self._save_data_summary(report, output_dir)
        
        # Save model information
        self._save_model_info(report, output_dir)
        
        # Save recommendations
        self._save_recommendations(report, output_dir)
        
        # Save comprehensive text report
        self._save_text_report(report, output_dir)
    
    def _save_data_summary(self, report, output_dir):
        """Save data summary to CSV."""
        filename = f"{output_dir}data_summary_{self.timestamp}.csv"
        
        summary_data = []
        for col, stats in report['data_summary'].items():
            row = {'Column': col, **stats}
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(filename, index=False)
        print(f"  ✓ Saved: {filename}")
    
    def _save_model_info(self, report, output_dir):
        """Save model information."""
        filename = f"{output_dir}model_info_{self.timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ARIMA MODEL INFORMATION\n")
            f.write("="*70 + "\n\n")
            
            model_info = report['model_information']
            f.write(f"ARIMA Order: {model_info.get('arima_order', 'N/A')}\n")
            f.write(f"Seasonal Order: {model_info.get('seasonal_order', 'N/A')}\n")
            f.write(f"AIC Score: {model_info.get('aic', 'N/A')}\n")
            f.write(f"BIC Score: {model_info.get('bic', 'N/A')}\n")
            f.write(f"Log Likelihood: {model_info.get('llf', 'N/A')}\n")
            f.write(f"Number of Observations: {model_info.get('nobs', 'N/A')}\n\n")
            
            if 'model_summary' in model_info:
                f.write(model_info['model_summary'])
        
        print(f"  ✓ Saved: {filename}")
    
    def _save_recommendations(self, report, output_dir):
        """Save recommendations."""
        filename = f"{output_dir}recommendations_{self.timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE RECOMMENDATIONS & ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            recs = report['recommendations']
            
            f.write("STRENGTHS\n")
            f.write("-"*70 + "\n")
            for item in recs.get('strengths', []):
                f.write(f"{item}\n")
            
            f.write("\nIMMEDIATE IMPROVEMENTS\n")
            f.write("-"*70 + "\n")
            for item in recs.get('improvements', []):
                f.write(f"{item}\n")
            
            f.write("\nFUTURE ENHANCEMENTS\n")
            f.write("-"*70 + "\n")
            for item in recs.get('enhancements', []):
                f.write(f"{item}\n")
            
            f.write("\nRISK WARNINGS\n")
            f.write("-"*70 + "\n")
            for item in recs.get('warnings', []):
                f.write(f"{item}\n")
        
        print(f"  ✓ Saved: {filename}")
    
    def _save_text_report(self, report, output_dir):
        """Save comprehensive text report."""
        filename = f"{output_dir}comprehensive_report_{self.timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ESG FORECASTING - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data Summary Section
            f.write("\n" + "="*70 + "\n")
            f.write("DATA SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            for col, stats in report['data_summary'].items():
                f.write(f"\n{col}:\n")
                f.write(f"  Observations: {stats['n_observations']}\n")
                f.write(f"  Original Missing: {stats['original_missing']}\n")
                f.write(f"  Fill Rate: {stats['fill_rate']:.2f}%\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std Dev: {stats['std']:.4f}\n")
                f.write(f"  Min: {stats['min']:.4f}\n")
                f.write(f"  Max: {stats['max']:.4f}\n")
                f.write(f"  Median: {stats['median']:.4f}\n")
            
            # Statistical Analysis
            f.write("\n" + "="*70 + "\n")
            f.write("STATISTICAL ANALYSIS\n")
            f.write("="*70 + "\n")
            
            stats_analysis = report['statistical_analysis']
            
            f.write("\nTrend Analysis:\n")
            trend = stats_analysis.get('overall_trend', {})
            f.write(f"  Direction: {trend.get('direction', 'N/A')}\n")
            f.write(f"  Change: {trend.get('change_percent', 0):.2f}%\n")
            
            f.write("\nVolatility Analysis:\n")
            volatility = stats_analysis.get('volatility', {})
            f.write(f"  Level: {volatility.get('level', 'N/A')}\n")
            f.write(f"  Coefficient of Variation: {volatility.get('coefficient_of_variation', 0):.2f}%\n")
            
            # Model Information
            f.write("\n" + "="*70 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("="*70 + "\n")
            
            model_info = report['model_information']
            f.write(f"ARIMA Order: {model_info.get('arima_order', 'N/A')}\n")
            f.write(f"AIC: {model_info.get('aic', 'N/A')}\n")
            f.write(f"BIC: {model_info.get('bic', 'N/A')}\n")
            
            # Recommendations
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n")
            
            recs = report['recommendations']
            for section, items in recs.items():
                f.write(f"\n{section.upper()}:\n")
                for item in items:
                    f.write(f"  {item}\n")
        
        print(f"  ✓ Saved: {filename}")
    
    def print_report_summary(self, report):
        """Print report summary to console."""
        print("\n" + "="*70)
        print("REPORT SUMMARY")
        print("="*70)
        
        print("\nDATA QUALITY:")
        for col, stats in report['data_summary'].items():
            print(f"\n  {col}:")
            print(f"    Fill Rate: {stats['fill_rate']:.1f}%")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Std Dev: {stats['std']:.2f}")
        
        print("\n\nRECOMMENDATIONS:")
        recs = report['recommendations']
        print("\n  STRENGTHS:")
        for item in recs['strengths']:
            print(f"    {item}")
        
        print("\n  IMPROVEMENTS:")
        for item in recs['improvements'][:3]:  # Show first 3
            print(f"    {item}")
        
        print("\n  WARNINGS:")
        for item in recs['warnings'][:2]:  # Show first 2
            print(f"    {item}")


if __name__ == "__main__":
    # Example usage
    from .data_generator import ESGDataGenerator
    from .data_processor import DataProcessor
    from .arima_model import ARIMAForecaster
    from .diagnostics import ModelDiagnostics
    
    generator = ESGDataGenerator()
    df = generator.generate()
    
    processor = DataProcessor(df)
    df_processed = processor.handle_missing_values(method='cubic')
    
    forecaster = ARIMAForecaster(df_processed['Environmental'])
    forecaster.auto_fit(verbose=False)
    forecaster.fit_arima(verbose=False)
    
    diagnostics = ModelDiagnostics(forecaster.results, df_processed['Environmental'])
    diagnostics.run_all_diagnostics(verbose=False)
    
    reporter = ReportGenerator(forecaster, df, df_processed, diagnostics)
    report = reporter.generate_full_report()
    reporter.print_report_summary(report)
