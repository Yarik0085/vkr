import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    def __init__(self):
        """Initialize FeatureEngineer"""
        pass
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional time features
        """
        # Create date features
        df['day_of_week'] = pd.to_datetime(df['year_dateotp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['year_dateotp']).dt.month
        df['quarter'] = pd.to_datetime(df['year_dateotp']).dt.quarter
        
        # Create season features
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 3, 7]) -> pd.DataFrame:
        """
        Create lag features for ticket sales
        
        Args:
            df (pd.DataFrame): Input dataframe
            lag_periods (List[int]): List of lag periods to create
            
        Returns:
            pd.DataFrame: Dataframe with lag features
        """
        # Sort by date
        df = df.sort_values('year_dateotp')
        
        # Create lag features for each target
        for target in ['target_count_tickets_lgot_255', 'target_count_tickets_lgot']:
            for lag in lag_periods:
                df[f'{target}_lag_{lag}'] = df[target].shift(lag)
                
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling statistics features
        
        Args:
            df (pd.DataFrame): Input dataframe
            windows (List[int]): List of window sizes for rolling statistics
            
        Returns:
            pd.DataFrame: Dataframe with rolling features
        """
        # Sort by date
        df = df.sort_values('year_dateotp')
        
        # Create rolling features for each target
        for target in ['target_count_tickets_lgot_255', 'target_count_tickets_lgot']:
            for window in windows:
                # Rolling mean
                df[f'{target}_rolling_mean_{window}'] = df[target].rolling(window=window).mean()
                # Rolling std
                df[f'{target}_rolling_std_{window}'] = df[target].rolling(window=window).std()
                # Rolling min
                df[f'{target}_rolling_min_{window}'] = df[target].rolling(window=window).min()
                # Rolling max
                df[f'{target}_rolling_max_{window}'] = df[target].rolling(window=window).max()
                
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between holidays and seasons
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        # Create holiday features if they don't exist
        if 'is_holyday' not in df.columns:
            df['is_holyday'] = 0
        if 'is_pre_holyday' not in df.columns:
            df['is_pre_holyday'] = 0
        if 'cat_pre_holyday' not in df.columns:
            df['cat_pre_holyday'] = 0
        if 'school_holyday' not in df.columns:
            df['school_holyday'] = 0
        
        # Holiday interactions
        df['is_holiday_summer'] = df['is_summer'] * df['is_holyday']
        df['is_holiday_winter'] = df['is_winter'] * df['is_holyday']
        df['is_holiday_spring'] = df['is_spring'] * df['is_holyday']
        df['is_holiday_autumn'] = df['is_autumn'] * df['is_holyday']
        
        # School holiday interactions
        df['is_school_holiday_summer'] = df['is_summer'] * df['school_holyday']
        df['is_school_holiday_winter'] = df['is_winter'] * df['school_holyday']
        df['is_school_holiday_spring'] = df['is_spring'] * df['school_holyday']
        df['is_school_holiday_autumn'] = df['is_autumn'] * df['school_holyday']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all additional features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all additional features
        """
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_interaction_features(df)
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        return df 