"""
Data Processor

Handles data cleaning, preprocessing, and missing value detection for ESG data.

Reference:
- Financial Time Series Forecasting Best Practices
"""

import pandas as pd
import numpy as np
from . import config


class DataProcessor:
    """
    Process and clean ESG data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with ESG data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.missing_summary = {}
        
    def analyze_missing_values(self):
        """
        Analyze missing values in the dataset.
        
        Returns
        -------
        dict
            Summary of missing values
        """
        print("\n" + "=" * 70)
        print("MISSING VALUES ANALYSIS")
        print("=" * 70)
        
        missing_info = {}
        total_cells = len(self.df)
        
        for col in self.df.columns:
            if col != 'Date':
                n_missing = self.df[col].isnull().sum()
                pct_missing = (n_missing / total_cells) * 100
                missing_info[col] = {
                    'count': n_missing,
                    'percentage': pct_missing,
                    'indices': self.df[self.df[col].isnull()].index.tolist()
                }
                
                print(f"\n{col}:")
                print(f"  Missing values: {n_missing}/{total_cells} ({pct_missing:.2f}%)")
                if n_missing > 0:
                    print(f"  Indices: {missing_info[col]['indices']}")
        
        self.missing_summary = missing_info
        return missing_info
    
    def get_data_quality_report(self):
        """
        Generate data quality report.
        
        Returns
        -------
        dict
            Data quality metrics
        """
        print("\n" + "=" * 70)
        print("DATA QUALITY REPORT")
        print("=" * 70)
        
        quality_report = {}
        
        for col in self.df.columns:
            if col != 'Date':
                n_missing = self.df[col].isnull().sum()
                data_points = len(self.df) - n_missing
                
                quality_report[col] = {
                    'total_rows': len(self.df),
                    'complete_rows': data_points,
                    'missing_rows': n_missing,
                    'completeness': (data_points / len(self.df)) * 100,
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'median': self.df[col].median(),
                }
                
                print(f"\n{col}:")
                print(f"  Completeness: {quality_report[col]['completeness']:.2f}%")
                print(f"  Mean: {quality_report[col]['mean']:.2f}")
                print(f"  Std Dev: {quality_report[col]['std']:.2f}")
                print(f"  Range: [{quality_report[col]['min']:.2f}, {quality_report[col]['max']:.2f}]")
        
        return quality_report
    
    def handle_missing_values(self, method='linear'):
        """
        Handle missing values using specified interpolation method.
        
        Parameters
        ----------
        method : str, default='linear'
            Interpolation method: 'linear', 'cubic', 'polynomial'
        
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled
        """
        print(f"\n" + "=" * 70)
        print(f"HANDLING MISSING VALUES - Method: {method.upper()}")
        print("=" * 70)
        
        from .interpolation import InterpolationHandler
        
        handler = InterpolationHandler(self.df)
        self.df = handler.interpolate(method=method)
        
        # Verify no missing values remain
        remaining_missing = self.df.isnull().sum().sum()
        if remaining_missing == 0:
            print(f"\n✓ All missing values successfully handled!")
        else:
            print(f"\n⚠ Warning: {remaining_missing} missing values remain")
        
        return self.df
    
    def normalize_data(self, method='minmax'):
        """
        Normalize data to [0, 1] range.
        
        Parameters
        ----------
        method : str, default='minmax'
            Normalization method: 'minmax', 'standard'
        
        Returns
        -------
        pd.DataFrame
            Normalized DataFrame
        """
        print(f"\n" + "=" * 70)
        print(f"NORMALIZING DATA - Method: {method.upper()}")
        print("=" * 70)
        
        df_normalized = self.df.copy()
        
        for col in df_normalized.columns:
            if col != 'Date':
                if method == 'minmax':
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                    print(f"\n{col}: MinMax normalized [{min_val:.2f}, {max_val:.2f}] → [0, 1]")
                
                elif method == 'standard':
                    mean = df_normalized[col].mean()
                    std = df_normalized[col].std()
                    df_normalized[col] = (df_normalized[col] - mean) / std
                    print(f"\n{col}: Standard normalized (μ={mean:.2f}, σ={std:.2f})")
        
        return df_normalized
    
    def get_summary_statistics(self):
        """
        Get comprehensive summary statistics.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        summary = self.df.describe().T
        print("\n", summary)
        
        return summary
    
    def export_to_csv(self, filepath):
        """
        Export processed data to CSV.
        
        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        self.df.to_csv(filepath, index=False)
        print(f"\n✓ Data exported to: {filepath}")
    
    def compare_before_after(self):
        """
        Compare original and processed data.
        
        Returns
        -------
        dict
            Comparison metrics
        """
        print("\n" + "=" * 70)
        print("BEFORE/AFTER COMPARISON")
        print("=" * 70)
        
        comparison = {}
        
        for col in self.df.columns:
            if col != 'Date':
                orig_missing = self.original_df[col].isnull().sum()
                proc_missing = self.df[col].isnull().sum()
                
                comparison[col] = {
                    'original_missing': orig_missing,
                    'processed_missing': proc_missing,
                    'filled': orig_missing - proc_missing,
                    'original_mean': self.original_df[col].mean(),
                    'processed_mean': self.df[col].mean(),
                }
                
                print(f"\n{col}:")
                print(f"  Original missing: {orig_missing}")
                print(f"  After processing: {proc_missing}")
                print(f"  Values filled: {comparison[col]['filled']}")
        
        return comparison


if __name__ == "__main__":
    # Example usage
    from .data_generator import ESGDataGenerator
    
    generator = ESGDataGenerator()
    df = generator.generate()
    
    processor = DataProcessor(df)
    processor.analyze_missing_values()
    processor.get_data_quality_report()
    processor.handle_missing_values(method='cubic')
    processor.get_summary_statistics()
    processor.compare_before_after()
