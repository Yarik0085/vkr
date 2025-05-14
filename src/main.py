import os
import pandas as pd
import numpy as np
import gc
from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from utils.visualization import Visualizer

def main():
    # Initialize components
    data_loader = DataLoader(data_dir='data', chunk_size=1000)  # Using smaller chunk size
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = data_loader.load_data()
    
    # Create features
    print("Creating features...")
    df = feature_engineer.create_all_features(df)
    
    # Prepare features for training
    print("Preparing features for training...")
    X, y, feature_names = data_loader.prepare_features(df)
    
    # Clean up memory
    del df
    gc.collect()
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    
    # Clean up memory
    del X, y
    gc.collect()
    
    # Train and evaluate models
    print("Training and evaluating models...")
    best_models, best_model_name = model_trainer.find_best_model(X_train, y_train, X_test, y_test)
    
    # Get feature importance
    print("Analyzing feature importance...")
    importance_df = model_trainer.get_feature_importance(best_models, feature_names)
    
    # Visualize results
    print("Generating visualizations...")
    visualizer.plot_feature_importance(importance_df)
    
    # Make predictions and plot for each target
    for i, target in enumerate(model_trainer.target_columns):
        y_pred = best_models[target].predict(X_test)
        y_true = y_test[:, i]
        
        visualizer.plot_predictions(y_true, y_pred, 
                                  title=f'{best_model_name} Predictions vs Actual - {target}')
        visualizer.plot_residuals(y_true, y_pred, 
                                title=f'{best_model_name} Residuals - {target}')
    
    # Plot correlation matrix
    visualizer.plot_correlation_matrix(pd.DataFrame(X_test, columns=feature_names))
    
    print("\nModel Performance Summary:")
    metrics = model_trainer.evaluate_model(best_models, X_test, y_test)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print(f"\nBest performing model: {best_model_name}")

if __name__ == "__main__":
    main() 