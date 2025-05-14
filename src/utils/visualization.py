import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    def __init__(self):
        """Initialize Visualizer"""
        plt.style.use('seaborn-v0_8')
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): DataFrame with feature importance
            top_n (int): Number of top features to plot
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Predictions vs Actual'):
        """
        Plot predictions against actual values
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title(title)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.tight_layout()
        plt.show()
        
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Residuals Plot'):
        """
        Plot residuals
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.show()
        
    def plot_sales_profile(self, df: pd.DataFrame, date_col: str, target_col: str, 
                         title: str = 'Sales Profile'):
        """
        Plot sales profile over time
        
        Args:
            df (pd.DataFrame): DataFrame with sales data
            date_col (str): Name of date column
            target_col (str): Name of target column
            title (str): Plot title
        """
        plt.figure(figsize=(15, 6))
        plt.plot(df[date_col], df[target_col])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = 'Correlation Matrix'):
        """
        Plot correlation matrix
        
        Args:
            df (pd.DataFrame): DataFrame to plot correlations for
            title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]], metric: str = 'rmse'):
        """
        Plot model comparison based on metrics
        
        Args:
            metrics_dict (Dict[str, Dict[str, float]]): Dictionary of model metrics
            metric (str): Metric to plot
        """
        models = list(metrics_dict.keys())
        values = [metrics_dict[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=values)
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Model')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 