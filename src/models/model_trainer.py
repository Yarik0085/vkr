import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self):
        """Initialize ModelTrainer"""
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42)
        }
        
        self.best_model = None
        self.best_model_name = None
        self.target_columns = ['target_count_tickets_lgot_255', 'target_count_tickets_lgot']
        
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a model for each target variable
        
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            
        Returns:
            Dict[str, Any]: Dictionary of trained models for each target
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        trained_models = {}
        for i, target in enumerate(self.target_columns):
            model = self.models[model_name]
            model.fit(X_train, y_train[:, i])
            trained_models[target] = model
            
        return trained_models
    
    def evaluate_model(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance for each target
        
        Args:
            models (Dict[str, Any]): Dictionary of trained models
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        metrics = {}
        
        for i, target in enumerate(self.target_columns):
            y_pred = models[target].predict(X_test)
            y_true = y_test[:, i]
            
            metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'{target}_mae'] = mean_absolute_error(y_true, y_pred)
            metrics[f'{target}_r2'] = r2_score(y_true, y_pred)
        
        # Calculate average metrics across targets
        metrics['avg_rmse'] = np.mean([metrics[f'{target}_rmse'] for target in self.target_columns])
        metrics['avg_mae'] = np.mean([metrics[f'{target}_mae'] for target in self.target_columns])
        metrics['avg_r2'] = np.mean([metrics[f'{target}_r2'] for target in self.target_columns])
        
        return metrics
    
    def find_best_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, Any], str]:
        """
        Find the best performing model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Tuple[Dict[str, Any], str]: Best models and model name
        """
        best_avg_rmse = float('inf')
        best_models = None
        best_model_name = None
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            trained_models = self.train_model(model_name, X_train, y_train)
            metrics = self.evaluate_model(trained_models, X_test, y_test)
            
            if metrics['avg_rmse'] < best_avg_rmse:
                best_avg_rmse = metrics['avg_rmse']
                best_models = trained_models
                best_model_name = model_name
                
        self.best_model = best_models
        self.best_model_name = best_model_name
        
        return best_models, best_model_name
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                               param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Optimize hyperparameters for each target
        
        Args:
            model_name (str): Name of the model to optimize
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            param_grid (Dict[str, List[Any]]): Parameter grid for optimization
            
        Returns:
            Dict[str, Any]: Dictionary of optimized models
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        optimized_models = {}
        for i, target in enumerate(self.target_columns):
            model = self.models[model_name]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train[:, i])
            optimized_models[target] = grid_search.best_estimator_
            
        return optimized_models
    
    def get_feature_importance(self, models: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from models
        
        Args:
            models (Dict[str, Any]): Dictionary of trained models
            feature_names (List[str]): List of feature names
            
        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        importance_dfs = []
        
        for target, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                raise ValueError(f"Model for {target} does not support feature importance")
                
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'target': target
            })
            importance_dfs.append(importance_df)
            
        # Combine importance from all targets
        combined_importance = pd.concat(importance_dfs)
        combined_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
        
        return combined_importance.sort_values('importance', ascending=False) 