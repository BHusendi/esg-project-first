"""
Visualization Module

Creates comprehensive plots and charts for ESG data analysis and forecasting.

Reference:
- matplotlib & seaborn best practices for financial visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import warnings

warnings.filterwarnings('ignore')

from . import config


class Visualizer:
    """
    Generate visualizations for ESG data and forecasting.
    
    Parameters
    ----------
    figsize : tuple, default=None
        Figure size (width, height)
    style : str, default=None
        Matplotlib style
    """
    
    def __init__(self, figsize=None, style=None):
        self.figsize = figsize or config.VISUALIZATION['figure_size']
        self.style = style or config.VISUALIZATION['style']
        self.colors = config.VISUALIZATION['colors']
        
        try:
            plt.style.use(self.style)
        except:
            pass
    
    def plot_interpolation_comparison(self, original_data, interpolated_data, 
                                     column_name, method, save_path=None):
        """
        Plot original vs interpolated data.
        
        Parameters
        ----------
        original_data : pd.Series
            Original data with missing values
        interpolated_data : pd.Series
            Interpolated data
        column_name : str
            Column name for title
        method : str
            Interpolation method used
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Original data
        ax1.plot(original_data.index, original_data.values, 
                marker='o', linestyle='-', color=self.colors['original'],
                label='Original (with missing)', linewidth=2)
        ax1.scatter(original_data[original_data.isnull()].index, 
                   [original_data.mean()]*len(original_data[original_data.isnull()]),
                   color='red', s=100, marker='x', label='Missing values', zorder=5)
        ax1.set_ylabel(column_name, fontsize=12, fontweight='bold')
        ax1.set_title(f'{column_name} - Original Data with Missing Values', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Interpolated data
        ax2.plot(interpolated_data.index, interpolated_data.values,
                marker='o', linestyle='-', color=self.colors['interpolated'],
                label=f'Interpolated ({method})', linewidth=2)
        ax2.fill_between(interpolated_data.index, interpolated_data.values,
                        alpha=0.2, color=self.colors['interpolated'])
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel(column_name, fontsize=12, fontweight='bold')
        ax2.set_title(f'{column_name} - After {method.capitalize()} Interpolation',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_arima_forecast(self, historical_data, forecast_data, 
                           forecast_index, confidence_intervals,
                           column_name, save_path=None):
        """
        Plot ARIMA forecast with confidence intervals.
        
        Parameters
        ----------
        historical_data : pd.Series
            Historical time series
        forecast_data : np.array
            Forecasted values
        forecast_index : pd.DatetimeIndex
            Forecast dates
        confidence_intervals : tuple
            Lower and upper confidence bounds
        column_name : str
            Column name for title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values,
               color=self.colors['original'], linewidth=2.5,
               label='Historical Data', marker='o', markersize=4)
        
        # Plot forecast
        ax.plot(forecast_index, forecast_data,
               color=self.colors['forecast'], linewidth=2.5,
               label='Forecast', marker='s', markersize=5, linestyle='--')
        
        # Plot confidence interval
        ax.fill_between(forecast_index,
                       confidence_intervals[0],
                       confidence_intervals[1],
                       alpha=0.25, color=self.colors['confidence'],
                       label='95% Confidence Interval')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel(column_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{column_name} - ARIMA Forecast with 95% Confidence Interval',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            print(f"✓ Forecast visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_residuals_diagnostics(self, residuals, save_path=None):
        """
        Plot residual diagnostics.
        
        Parameters
        ----------
        residuals : pd.Series or np.array
            Model residuals
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals, color='steelblue', linewidth=1.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        axes[0, 1].hist(residuals, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Histogram of Residuals', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Residuals', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ACF-like plot (autocorrelation)
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=20, ax=axes[1, 1], title='ACF Plot')
        axes[1, 1].set_title('Autocorrelation of Residuals', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            print(f"✓ Diagnostics plot saved to: {save_path}")
        
        plt.show()
    
    def plot_data_summary(self, df, column_names, save_path=None):
        """
        Plot summary statistics for all columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to plot
        column_names : list
            Columns to plot
        save_path : str, optional
            Path to save figure
        """
        n_cols = len(column_names)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4*n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(column_names):
            axes[idx].plot(df.index, df[col], marker='o', linewidth=2,
                          color=self.colors['original'], markersize=5)
            axes[idx].fill_between(df.index, df[col], alpha=0.2,
                                   color=self.colors['original'])
            axes[idx].set_ylabel(col, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{col} Time Series', fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            if idx == n_cols - 1:
                axes[idx].set_xlabel('Date', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            print(f"✓ Summary plot saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from .data_generator import ESGDataGenerator
    from .data_processor import DataProcessor
    
    generator = ESGDataGenerator()
    df = generator.generate()
    
    processor = DataProcessor(df)
    df_processed = processor.handle_missing_values(method='cubic')
    
    visualizer = Visualizer()
    visualizer.plot_interpolation_comparison(
        df['Environmental'],
        df_processed['Environmental'],
        'Environmental Score',
        'cubic'
    )
