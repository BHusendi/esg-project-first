"""
Main Script: Complete ESG Forecasting Pipeline

This script runs the complete ESG forecasting workflow:
1. Data generation or loading
2. Data preprocessing and interpolation
3. ARIMA modeling
4. Forecasting with diagnostics
5. Comprehensive report generation

Usage:
    python scripts/04_full_analysis.py
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import ESGDataGenerator
from src.data_processor import DataProcessor
from src.arima_model import ARIMAForecaster
from src.diagnostics import ModelDiagnostics
from src.visualization import Visualizer
from src.reporting import ReportGenerator
from src import config


def main():
    """Execute complete ESG forecasting pipeline."""
    
    print("\n" + "="*70)
    print(" "*15 + "ESG FORECASTING WITH ARIMA & INTERPOLATION")
    print(" "*20 + "Complete Analysis Pipeline v1.0")
    print("="*70)
    print(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================================================
    # STEP 1: DATA GENERATION
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 1: DATA GENERATION & LOADING")
    print("#"*70)
    
    generator = ESGDataGenerator(
        n_periods=config.DATA_GENERATION['n_periods'],
        missing_rate=config.DATA_GENERATION['missing_rate']
    )
    df_original = generator.generate()
    
    # Create output directories
    os.makedirs(config.PATHS['data_raw'], exist_ok=True)
    os.makedirs(config.PATHS['data_processed'], exist_ok=True)
    os.makedirs(config.PATHS['reports'], exist_ok=True)
    os.makedirs(config.PATHS['visualizations'], exist_ok=True)
    
    # Save original data
    df_original.to_csv(config.PATHS['sample_data'], index=False)
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 2: DATA PREPROCESSING & MISSING VALUES ANALYSIS")
    print("#"*70)
    
    processor = DataProcessor(df_original)
    processor.analyze_missing_values()
    processor.get_data_quality_report()
    processor.compare_before_after()
    
    # ========================================================================
    # STEP 3: INTERPOLATION
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 3: INTERPOLATION & GAP FILLING")
    print("#"*70)
    
    interpolation_method = config.INTERPOLATION['method']
    df_processed = processor.handle_missing_values(method=interpolation_method)
    
    # Summary statistics after processing
    processor.get_summary_statistics()
    
    # ========================================================================
    # STEP 4: VISUALIZATION - Before/After Interpolation
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 4: VISUALIZATION - INTERPOLATION COMPARISON")
    print("#"*70)
    
    visualizer = Visualizer()
    
    # Plot each ESG component
    for col in ['Environmental', 'Social', 'Governance']:
        if col in df_original.columns and col in df_processed.columns:
            save_path = os.path.join(
                config.PATHS['visualizations'],
                f"01_interpolation_{col.lower()}.png"
            )
            visualizer.plot_interpolation_comparison(
                df_original[col],
                df_processed[col],
                col,
                interpolation_method,
                save_path=save_path
            )
    
    # ========================================================================
    # STEP 5: ARIMA MODELING - ENVIRONMENTAL PARAMETER
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 5: ARIMA MODELING - ENVIRONMENTAL PARAMETER")
    print("#"*70)
    
    # Test stationarity
    forecaster_env = ARIMAForecaster(df_processed['Environmental'], name='Environmental')
    stationarity = forecaster_env.test_stationarity()
    
    # Auto-fit ARIMA parameters
    forecaster_env.auto_fit()
    
    # Fit the model
    forecaster_env.fit_arima()
    
    # ========================================================================
    # STEP 6: MODEL DIAGNOSTICS
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 6: MODEL DIAGNOSTICS & VALIDATION")
    print("#"*70)
    
    diagnostics = ModelDiagnostics(forecaster_env.results, df_processed['Environmental'])
    diagnostics.run_all_diagnostics()
    
    # ========================================================================
    # STEP 7: FORECASTING
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 7: FORECASTING & PREDICTIONS")
    print("#"*70)
    
    forecast_result = forecaster_env.forecast(
        steps=config.FORECASTING['forecast_periods'],
        confidence=config.FORECASTING['confidence_level']
    )
    
    # Create forecast dates
    import pandas as pd
    # Ambil tanggal terakhir dengan benar
    if 'Date' in df_processed.columns:
        last_date = pd.to_datetime(df_processed['Date'].iloc[-1])
    else:
        last_date = pd.to_datetime(df_processed.index[-1])

    # Buat forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=config.FORECASTING['forecast_periods'],
        freq='M'
    )
    
    # ========================================================================
    # STEP 8: FORECAST VISUALIZATION
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 8: VISUALIZATION - ARIMA FORECAST")
    print("#"*70)
    
    save_path = os.path.join(config.PATHS['visualizations'], "02_arima_forecast.png")
    visualizer.plot_arima_forecast(
        df_processed['Environmental'],
        forecast_result['forecast'].values,
        forecast_dates,
        (forecast_result['confidence_intervals'].iloc[:, 0].values,
         forecast_result['confidence_intervals'].iloc[:, 1].values),
        'Environmental Score',
        save_path=save_path
    )
    
    # ========================================================================
    # STEP 9: RESIDUAL DIAGNOSTICS PLOT
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 9: VISUALIZATION - RESIDUAL DIAGNOSTICS")
    print("#"*70)
    
    save_path = os.path.join(config.PATHS['visualizations'], "03_residual_diagnostics.png")
    visualizer.plot_residuals_diagnostics(
        forecaster_env.results.resid,
        save_path=save_path
    )
    
    # ========================================================================
    # STEP 10: COMPREHENSIVE REPORTING
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 10: COMPREHENSIVE REPORT GENERATION")
    print("#"*70)
    
    reporter = ReportGenerator(
        forecaster_env,
        df_original,
        df_processed,
        diagnostics
    )
    
    report = reporter.generate_full_report(
        output_dir=config.PATHS['reports'],
        save_files=True
    )
    
    # Print summary
    reporter.print_report_summary(report)
    
    # ========================================================================
    # STEP 11: EXPORT FORECAST RESULTS
    # ========================================================================
    print("\n" + "#"*70)
    print("# STEP 11: EXPORTING FORECAST RESULTS")
    print("#"*70)
    
    forecast_df = forecast_result['forecast_df'].copy()
    forecast_csv = os.path.join(config.PATHS['reports'], f"forecast_results_{reporter.timestamp}.csv")
    forecast_df.to_csv(forecast_csv)
    print(f"\n✓ Forecast results saved to: {forecast_csv}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nGenerated Files:")
    print(f"  Reports: {config.PATHS['reports']}")
    print(f"  Visualizations: {config.PATHS['visualizations']}")
    print(f"  Data: {config.PATHS['data_processed']}")
    print("\nNext Steps:")
    print("  1. Review generated reports in 'outputs/reports/'")
    print("  2. Check visualizations in 'outputs/visualizations/'")
    print("  3. Implement recommendations from the report")
    print("  4. Monitor forecast vs actual values")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
