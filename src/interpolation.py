"""
Interpolation Methods

Implements various interpolation techniques for handling missing values in time series.

Reference:
- Interpolation in Environmental Science: ScienceDirect
- Financial Time Series Analysis Best Practices
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from . import config


class InterpolationHandler:
    """
    Handle missing values using various interpolation methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
    
    def interpolate(self, method='linear'):
        """
        Interpolate missing values using specified method.
        
        Parameters
        ----------
        method : str, default='linear'
            Interpolation method: 'linear', 'cubic', 'polynomial'
        
        Returns
        -------
        pd.DataFrame
            DataFrame with interpolated values
        """
        print(f"\n{'='*70}")
        print(f"INTERPOLATION - Method: {method.upper()}")
        print(f"{'='*70}")
        
        df_interpolated = self.df.copy()
        
        for col in df_interpolated.columns:
            if col != 'Date':
                n_missing_before = df_interpolated[col].isnull().sum()
                
                if method == 'linear':
                    df_interpolated[col] = self._linear_interpolate(df_interpolated[col])
                
                elif method == 'cubic':
                    df_interpolated[col] = self._cubic_interpolate(df_interpolated[col])
                
                elif method == 'polynomial':
                    df_interpolated[col] = self._polynomial_interpolate(df_interpolated[col])
                
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
                
                n_missing_after = df_interpolated[col].isnull().sum()
                n_filled = n_missing_before - n_missing_after
                
                print(f"\n{col}:")
                print(f"  Missing before: {n_missing_before}")
                print(f"  Missing after: {n_missing_after}")
                print(f"  Values filled: {n_filled}")
        
        return df_interpolated
    
    def _linear_interpolate(self, series):
        """
        Linear interpolation.
        
        Connects missing points with straight lines.
        Best for: Regular, linear patterns.
        
        Parameters
        ----------
        series : pd.Series
            Series with missing values
        
        Returns
        -------
        pd.Series
            Interpolated series
        """
        return series.interpolate(method='linear', limit_direction='both')
    
    def _cubic_interpolate(self, series):
        """
        Cubic spline interpolation.
        
        Smooth curve fitting using cubic polynomials.
        Best for: Smooth, curved patterns.
        
        Parameters
        ----------
        series : pd.Series
            Series with missing values
        
        Returns
        -------
        pd.Series
            Interpolated series
        """
        return series.interpolate(method='cubic', limit_direction='both')
    
    def _polynomial_interpolate(self, series, order=None):
        """
        Polynomial interpolation.
        
        Higher-order polynomial fitting.
        Best for: Complex patterns.
        
        Parameters
        ----------
        series : pd.Series
            Series with missing values
        order : int, default=None
            Polynomial order (default: 3)
        
        Returns
        -------
        pd.Series
            Interpolated series
        """
        if order is None:
            order = config.INTERPOLATION['polynomial_order']
        
        return series.interpolate(method='polynomial', order=order, limit_direction='both')
    
    def compare_interpolation_methods(self):
        """
        Compare different interpolation methods.
        
        Returns
        -------
        dict
            Comparison results
        """
        print(f"\n{'='*70}")
        print("COMPARING INTERPOLATION METHODS")
        print(f"{'='*70}")
        
        methods = ['linear', 'cubic', 'polynomial']
        results = {}
        
        for method in methods:
            df_interp = self.interpolate(method=method)
            results[method] = df_interp
        
        return results
    
    def get_interpolation_statistics(self, df_interpolated):
        """
        Get statistics about interpolation.
        
        Parameters
        ----------
        df_interpolated : pd.DataFrame
            Interpolated DataFrame
        
        Returns
        -------
        dict
            Interpolation statistics
        """
        stats = {}
        
        for col in self.df.columns:
            if col != 'Date':
                # Find interpolated values
                original_missing = self.original_df[col].isnull()
                interpolated_values = df_interpolated.loc[original_missing, col]
                
                if len(interpolated_values) > 0:
                    stats[col] = {
                        'count': len(interpolated_values),
                        'mean_of_filled': interpolated_values.mean(),
                        'std_of_filled': interpolated_values.std(),
                        'min_filled': interpolated_values.min(),
                        'max_filled': interpolated_values.max(),
                    }
        
        return stats


if __name__ == "__main__":
    # Example usage
    from .data_generator import ESGDataGenerator
    
    generator = ESGDataGenerator()
    df = generator.generate()
    
    handler = InterpolationHandler(df)
    df_linear = handler.interpolate(method='linear')
    df_cubic = handler.interpolate(method='cubic')
    df_poly = handler.interpolate(method='polynomial')
    
    print("\n✓ Interpolation completed for all methods")
