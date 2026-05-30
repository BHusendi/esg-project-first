"""
ESG Data Generator

Generates synthetic ESG (Environmental, Social, Governance) data for testing
and demonstration purposes.

Reference:
- S&P 500 ESG Index Prediction with LSTM and ARIMA Model (ICEMESS 2023)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from . import config


class ESGDataGenerator:
    """
    Generates synthetic ESG time-series data.
    
    Parameters
    ----------
    n_periods : int, default=36
        Number of periods to generate (months)
    start_date : str, default='2021-01-01'
        Start date for the time series
    missing_rate : float, default=0.15
        Percentage of missing values (0.0-1.0)
    seed : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, n_periods=None, start_date=None, missing_rate=None, seed=None):
        self.n_periods = n_periods or config.DATA_GENERATION['n_periods']
        self.start_date = start_date or config.DATA_GENERATION['start_date']
        self.missing_rate = missing_rate or config.DATA_GENERATION['missing_rate']
        self.seed = seed or config.DATA_GENERATION['seed']
        
        np.random.seed(self.seed)
    
    def generate(self):
        """
        Generate synthetic ESG data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Date, Environmental, Social, Governance
        """
        print("=" * 70)
        print("GENERATING SYNTHETIC ESG DATA")
        print("=" * 70)
        
        # Create date range
        dates = pd.date_range(start=self.start_date, periods=self.n_periods, freq='M')
        
        # Generate trend components
        t = np.arange(self.n_periods)
        
        # Environmental: upward trend + seasonality + noise
        env_trend = 60 + 0.5 * t
        env_seasonal = 3 * np.sin(2 * np.pi * t / 12)
        env_noise = np.random.normal(0, 2, self.n_periods)
        environmental = env_trend + env_seasonal + env_noise
        
        # Social: steady with slight variations
        social_trend = 65 + 0.3 * t
        social_seasonal = 2 * np.cos(2 * np.pi * t / 12)
        social_noise = np.random.normal(0, 2, self.n_periods)
        social = social_trend + social_seasonal + social_noise
        
        # Governance: stable with minor fluctuations
        governance_trend = 70 + 0.2 * t
        governance_noise = np.random.normal(0, 2.5, self.n_periods)
        governance = governance_trend + governance_noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Environmental': environmental,
            'Social': social,
            'Governance': governance
        })
        
        # Introduce missing values randomly
        self._introduce_missing_values(df)
        
        # Clip values to realistic range (0-100)
        for col in ['Environmental', 'Social', 'Governance']:
            df[col] = df[col].clip(0, 100)
        
        print(f"\n✓ Generated {self.n_periods} periods of data")
        print(f"  Start Date: {df['Date'].iloc[0].strftime('%Y-%m-%d')}")
        print(f"  End Date: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"  Missing Rate: {self.missing_rate * 100:.1f}%")
        print(f"\nDataFrame Info:")
        print(df.info())
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    
    def _introduce_missing_values(self, df):
        """
        Randomly introduce missing values into the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to modify
        """
        columns = ['Environmental', 'Social', 'Governance']
        n_missing = int(self.n_periods * self.missing_rate)
        
        for col in columns:
            # Randomly select indices for missing values
            missing_indices = np.random.choice(
                self.n_periods, 
                size=n_missing, 
                replace=False
            )
            df.loc[missing_indices, col] = np.nan
    
    def save_data(self, filepath):
        """
        Save generated data to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save the CSV file
        """
        df = self.generate()
        df.to_csv(filepath, index=False)
        print(f"\n✓ Data saved to: {filepath}")
        return df


if __name__ == "__main__":
    # Example usage
    generator = ESGDataGenerator()
    df = generator.generate()
    
    # Save to CSV
    import os
    os.makedirs(config.PATHS['data_raw'], exist_ok=True)
    generator.save_data(config.PATHS['sample_data'])
