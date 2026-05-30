"""
Configuration settings for ESG Forecasting project.

Modify these settings to customize the behavior of the application.
"""

# ============================================================================
# DATA GENERATION SETTINGS
# ============================================================================
DATA_GENERATION = {
    'n_periods': 36,                    # Number of monthly periods (3 years)
    'start_date': '2021-01-01',        # Start date
    'frequency': 'M',                  # Monthly frequency
    'seed': 42,                        # Random seed for reproducibility
    'missing_rate': 0.15,              # Percentage of missing values
}

# ============================================================================
# INTERPOLATION SETTINGS
# ============================================================================
INTERPOLATION = {
    'method': 'cubic',                 # Options: 'linear', 'cubic', 'polynomial'
    'polynomial_order': 3,             # Order for polynomial interpolation
    'limit_direction': 'both',         # Forward fill direction
    'limit_area': 'inside',            # Fill inside only, not edges
}

# ============================================================================
# ARIMA SETTINGS
# ============================================================================
ARIMA = {
    'auto_fit': True,                  # Use auto_arima for parameter selection
    'max_p': 5,                        # Max AR order to search
    'max_d': 2,                        # Max differencing to search
    'max_q': 5,                        # Max MA order to search
    'max_seasonal_p': 2,               # Max seasonal AR
    'max_seasonal_d': 1,               # Max seasonal differencing
    'max_seasonal_q': 2,               # Max seasonal MA
    'seasonal_period': 12,             # 12 months for yearly seasonality
    'stepwise': True,                  # Use stepwise selection
    'trace': False,                    # Show fitting progress
}

# ============================================================================
# FORECASTING SETTINGS
# ============================================================================
FORECASTING = {
    'forecast_periods': 12,            # Number of periods to forecast (12 months)
    'confidence_level': 0.95,          # Confidence interval level
}

# ============================================================================
# PATHS & OUTPUTS
# ============================================================================
PATHS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'reports': 'outputs/reports/',
    'visualizations': 'outputs/visualizations/',
    'sample_data': 'data/raw/sample_esg_data.csv',
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
VISUALIZATION = {
    'figure_size': (14, 8),            # Figure size (width, height)
    'dpi': 100,                        # Resolution (dots per inch)
    'style': 'seaborn-v0_8',           # Plotting style
    'colors': {
        'original': '#1f77b4',         # Blue
        'interpolated': '#ff7f0e',     # Orange
        'forecast': '#2ca02c',         # Green
        'confidence': '#d62728',       # Red
    },
}

# ============================================================================
# STATISTICAL TESTS
# ============================================================================
STATISTICAL = {
    'adf_significance': 0.05,          # ADF test significance level
    'ljung_box_lags': 10,              # Lags for Ljung-Box test
    'normality_test': 'shapiro',       # Options: 'shapiro', 'normaltest'
}

# ============================================================================
# REPORTING SETTINGS
# ============================================================================
REPORTING = {
    'generate_csv': True,              # Export to CSV
    'generate_plots': True,            # Generate visualization plots
    'verbose': True,                   # Print detailed output
    'decimal_places': 4,               # Decimal places for output
    'timestamp_reports': True,         # Add timestamp to report filenames
}

# ============================================================================
# Feature Recommendations Thresholds
# ============================================================================
RECOMMENDATIONS_THRESHOLDS = {
    'rmse_good': 2.0,                  # RMSE threshold for "good" fit
    'rmse_acceptable': 4.0,            # RMSE threshold for "acceptable"
    'r2_good': 0.85,                   # R² threshold for "good" fit
    'r2_acceptable': 0.70,             # R² threshold for "acceptable"
    'missing_rate_good': 0.05,         # Good missing data rate (5%)
    'missing_rate_acceptable': 0.10,   # Acceptable missing data rate (10%)
    'ljung_box_pvalue': 0.05,          # Ljung-Box p-value threshold
}
