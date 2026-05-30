"""
Model Diagnostics

Performs comprehensive model validation and diagnostics for ARIMA models.

Reference:
- statsmodels ARIMA Diagnostics Documentation
- Time Series Analysis Best Practices
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
from . import config


class ModelDiagnostics:
    """
    Diagnose and validate fitted ARIMA models.
    
    Parameters
    ----------
    results : statsmodels result object
        Fitted ARIMA model results
    timeseries : pd.Series
        Original time series data
    """
    
    def __init__(self, results, timeseries):
        self.results = results
        self.timeseries = timeseries
        self.residuals = results.resid
        self.diagnostics = {}
    
    def run_all_diagnostics(self, verbose=True):
        """
        Run all diagnostic tests.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print results
        
        Returns
        -------
        dict
            All diagnostic results
        """
        if verbose:
            print(f"\n{'='*70}")
            print("MODEL DIAGNOSTICS")
            print(f"{'='*70}")
        
        # Run all tests
        residual_stats = self.residual_analysis(verbose=verbose)
        acf_test = self.autocorrelation_test(verbose=verbose)
        normality = self.normality_test(verbose=verbose)
        heteroscedasticity = self.heteroscedasticity_test(verbose=verbose)
        
        self.diagnostics = {
            'residual_statistics': residual_stats,
            'autocorrelation': acf_test,
            'normality': normality,
            'heteroscedasticity': heteroscedasticity
        }
        
        return self.diagnostics
    
    def residual_analysis(self, verbose=True):
        """
        Analyze residuals from the fitted model.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print results
        
        Returns
        -------
        dict
            Residual statistics
        """
        if verbose:
            print(f"\n{'-'*70}")
            print("RESIDUAL ANALYSIS")
            print(f"{'-'*70}")
        
        residuals = self.residuals
        
        stats_dict = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
        }
        
        if verbose:
            print(f"\nMean: {stats_dict['mean']:.6f}")
            print(f"Std Dev: {stats_dict['std']:.6f}")
            print(f"Min: {stats_dict['min']:.6f}")
            print(f"Max: {stats_dict['max']:.6f}")
            print(f"Skewness: {stats_dict['skewness']:.6f}")
            print(f"Kurtosis: {stats_dict['kurtosis']:.6f}")
            
            # Check if mean is close to zero (ideal)
            if abs(stats_dict['mean']) < 0.1 * stats_dict['std']:
                print(f"\n✓ Mean is close to zero (ideal for residuals)")
            else:
                print(f"\n✗ Mean is significantly different from zero")
        
        return stats_dict
    
    def autocorrelation_test(self, lags=None, verbose=True):
        """
        Test for autocorrelation in residuals (Ljung-Box test).
        
        Parameters
        ----------
        lags : int, default=None
            Number of lags to test
        verbose : bool, default=True
            Print results
        
        Returns
        -------
        dict
            Test results
        """
        if lags is None:
            lags = config.STATISTICAL['ljung_box_lags']
        
        if verbose:
            print(f"\n{'-'*70}")
            print("AUTOCORRELATION TEST (Ljung-Box)")
            print(f"{'-'*70}")
        
        # Manual Ljung-Box calculation
        residuals = self.residuals
        n = len(residuals)
        
        # ACF of residuals
        acf_vals = acf(residuals, nlags=lags)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum([acf_vals[i]**2 / (n - i) for i in range(1, lags + 1)])
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        test_dict = {
            'test_statistic': lb_stat,
            'p_value': p_value,
            'lags': lags,
            'has_autocorrelation': p_value < 0.05
        }
        
        if verbose:
            print(f"\nLjung-Box Test Statistic: {lb_stat:.6f}")
            print(f"P-value: {p_value:.6f}")
            print(f"Lags: {lags}")
            
            if p_value < 0.05:
                print(f"\n✗ Significant autocorrelation detected (p < 0.05)")
                print(f"  Residuals are not white noise")
            else:
                print(f"\n✓ No significant autocorrelation (p > 0.05)")
                print(f"  Residuals appear to be white noise")
        
        return test_dict
    
    def normality_test(self, test_type='shapiro', verbose=True):
        """
        Test for normality of residuals.
        
        Parameters
        ----------
        test_type : str, default='shapiro'
            Test type: 'shapiro' or 'normaltest'
        verbose : bool, default=True
            Print results
        
        Returns
        -------
        dict
            Test results
        """
        if verbose:
            print(f"\n{'-'*70}")
            print(f"NORMALITY TEST ({test_type.upper()})")
            print(f"{'-'*70}")
        
        residuals = self.residuals
        
        if test_type == 'shapiro':
            stat, p_value = stats.shapiro(residuals)
            test_name = "Shapiro-Wilk Test"
        else:  # normaltest
            stat, p_value = stats.normaltest(residuals)
            test_name = "Anderson-Darling Test"
        
        test_dict = {
            'test_name': test_name,
            'test_statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        if verbose:
            print(f"\n{test_name}:")
            print(f"Test Statistic: {stat:.6f}")
            print(f"P-value: {p_value:.6f}")
            
            if p_value > 0.05:
                print(f"\n✓ Residuals appear to be normally distributed (p > 0.05)")
            else:
                print(f"\n✗ Residuals do not appear to be normally distributed (p < 0.05)")
        
        return test_dict
    
    def heteroscedasticity_test(self, verbose=True):
        """
        Test for heteroscedasticity in residuals.
        
        Checks if variance is constant over time.
        
        Parameters
        ----------
        verbose : bool, default=True
            Print results
        
        Returns
        -------
        dict
            Test results
        """
        if verbose:
            print(f"\n{'-'*70}")
            print("HETEROSCEDASTICITY TEST")
            print(f"{'-'*70}")
        
        residuals = self.residuals
        
        # Split residuals into two halves
        n = len(residuals)
        first_half = residuals[:n//2]
        second_half = residuals[n//2:]
        
        # Levene's test for equal variances
        stat, p_value = stats.levene(first_half, second_half)
        
        test_dict = {
            'test_statistic': stat,
            'p_value': p_value,
            'has_heteroscedasticity': p_value < 0.05,
            'var_first_half': first_half.var(),
            'var_second_half': second_half.var(),
        }
        
        if verbose:
            print(f"\nLevene's Test (for equal variances):")
            print(f"Test Statistic: {stat:.6f}")
            print(f"P-value: {p_value:.6f}")
            print(f"\nVariance (first half): {test_dict['var_first_half']:.6f}")
            print(f"Variance (second half): {test_dict['var_second_half']:.6f}")
            
            if p_value > 0.05:
                print(f"\n✓ Constant variance assumed (homoscedasticity)")
            else:
                print(f"\n✗ Variance is not constant (heteroscedasticity)")
        
        return test_dict
    
    def get_diagnostics_summary(self):
        """
        Get summary of all diagnostics.
        
        Returns
        -------
        dict
            Summary of diagnostics
        """
        return self.diagnostics


if __name__ == "__main__":
    # Example usage
    from .data_generator import ESGDataGenerator
    from .data_processor import DataProcessor
    from .arima_model import ARIMAForecaster
    
    # Generate and process data
    generator = ESGDataGenerator()
    df = generator.generate()
    
    processor = DataProcessor(df)
    df = processor.handle_missing_values(method='cubic')
    
    # Fit ARIMA model
    forecaster = ARIMAForecaster(df['Environmental'], name='Environmental')
    forecaster.auto_fit(verbose=False)
    forecaster.fit_arima(verbose=False)
    
    # Run diagnostics
    diagnostics = ModelDiagnostics(forecaster.results, df['Environmental'])
    diagnostics.run_all_diagnostics()
