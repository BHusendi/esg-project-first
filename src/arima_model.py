"""
ARIMA Model

Implements ARIMA (AutoRegressive Integrated Moving Average) modeling for
time-series forecasting of ESG parameters.

Reference:
- S&P 500 ESG Index Prediction with LSTM and ARIMA Model (ICEMESS 2023)
- statsmodels ARIMA/SARIMAX Documentation
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA model for time-series forecasting.
    
    Parameters
    ----------
    timeseries : pd.Series or np.array
        Time series data
    name : str, default='ESG'
        Name of the series
    """
    
    def __init__(self, timeseries, name='ESG'):
        self.series = timeseries
        self.name = name
        self.model = None
        self.results = None
        self.order = None
        self.seasonal_order = None
    
    def test_stationarity(self, verbose=True):
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STATIONARITY TEST - Augmented Dickey-Fuller (ADF)")
            print(f"{'='*70}")
        
        result = adfuller(self.series.dropna())
        
        if verbose:
            print(f"\nADF Statistic: {result[0]:.6f}")
            print(f"P-value: {result[1]:.6f}")
            print(f"# of Lags: {result[2]}")
            print(f"# of Observations: {result[3]}")
            print(f"Critical Values:")
            for key, value in result[4].items():
                print(f"  {key}: {value:.3f}")
            
            if result[1] <= 0.05:
                print(f"\n✓ Series is STATIONARY (p-value = {result[1]:.6f})")
            else:
                print(f"\n✗ Series is NON-STATIONARY (p-value = {result[1]:.6f})")
                print(f"  Recommendation: Differencing needed (d > 0)")
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def auto_fit(self, verbose=True):
        """
        Automatically find optimal ARIMA parameters using grid search.
        """
        if verbose:
            print(f"\n{'='*70}")
            print("AUTO-FIT ARIMA PARAMETERS (Grid Search)")
            print(f"{'='*70}")
        
        best_aic = float('inf')
        best_order = None
        
        # Grid search parameters (p, d, q)
        p_range = range(0, 4)
        d_range = range(0, 3)
        q_range = range(0, 4)
        
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = SARIMAX(
                            self.series.dropna(),
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False)
                        
                        if verbose:
                            print(f"  Testing ARIMA{(p,d,q)}: AIC={results.aic:.2f}")
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            self.results = results
                            
                    except Exception as e:
                        continue
        
        self.order = best_order
        
        if verbose:
            print(f"\n✓ Optimal ARIMA Order: {self.order}")
            print(f"  Best AIC: {best_aic:.2f}")
            if self.results:
                print(f"  BIC: {self.results.bic:.2f}")
        
        return self.order
    
    def fit_arima(self, order=None, verbose=True):
        """
        Fit ARIMA model with specified parameters.
        """
        if verbose:
            print(f"\n{'='*70}")
            print("FITTING ARIMA MODEL")
            print(f"{'='*70}")
        
        if order is None:
            order = self.order or (1, 1, 1)
        
        self.model = SARIMAX(
            self.series,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.results = self.model.fit(disp=False)
        
        if verbose:
            print(f"\nARIMA Order: {order}")
            print(f"\nAIC: {self.results.aic:.2f}")
            print(f"BIC: {self.results.bic:.2f}")
        
        return self
    
    def forecast(self, steps=12, confidence=0.95):
        """
        Generate forecast using fitted ARIMA model.
        """
        if self.results is None:
            raise ValueError("Model must be fitted first. Call fit_arima() or auto_fit()")
        
        print(f"\n{'='*70}")
        print("FORECASTING")
        print(f"{'='*70}")
        print(f"\nForecast Periods: {steps}")
        print(f"Confidence Level: {confidence*100:.0f}%")
        
        # Generate forecast
        forecast_result = self.results.get_forecast(steps=steps)
        forecast_df = forecast_result.conf_int(alpha=1-confidence)
        forecast_df['forecast'] = forecast_result.predicted_mean
        forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
        forecast_df = forecast_df[['forecast', 'lower_ci', 'upper_ci']]
        
        print(f"\nForecast Results (first 5 rows):")
        print(forecast_df.head())
        
        return {
            'forecast': forecast_result.predicted_mean,
            'confidence_intervals': forecast_result.conf_int(alpha=1-confidence),
            'forecast_df': forecast_df,
            'steps': steps,
            'confidence': confidence
        }


if __name__ == "__main__":
    print("ARIMA Forecaster module loaded successfully")