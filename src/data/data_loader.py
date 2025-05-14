import os
import pandas as pd
import numpy as np
import gc
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_dir: str, chunk_size: int = 1000):
        """
        Initialize DataLoader with path to data directory
        
        Args:
            data_dir (str): Path to directory containing data subdirectories
            chunk_size (int): Number of rows to process at once
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def load_and_process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of data
        
        Args:
            chunk (pd.DataFrame): Input data chunk
            
        Returns:
            pd.DataFrame: Processed chunk
        """
        # Handle missing values
        chunk = chunk.ffill()
        
        # Convert date columns
        date_columns = ['year_dateotp', 'month_dateotp']
        for col in date_columns:
            if col in chunk.columns:
                try:
                    chunk[col] = pd.to_datetime(chunk[col], format='%Y-%m-%d')
                except ValueError:
                    try:
                        chunk[col] = pd.to_datetime(chunk[col], format='%Y')
                    except ValueError:
                        chunk[col] = pd.to_datetime(chunk[col])
        
        # Encode categorical variables
        categorical_columns = ['season', 'dow_dateotp']
        for col in categorical_columns:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype('category')
        
        return chunk
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and combine all CSV files from data subdirectories
        
        Returns:
            pd.DataFrame: Combined dataframe
        """
        processed_chunks = []
        
        # Get all pzd_* directories
        pzd_dirs = [d for d in os.listdir(self.data_dir) if d.startswith('pzd_')]
        
        for pzd_dir in pzd_dirs:
            pzd_path = os.path.join(self.data_dir, pzd_dir)
            if os.path.isdir(pzd_path):
                # Get all CSV files in the pzd directory
                csv_files = [f for f in os.listdir(pzd_path) if f.endswith('.csv')]
                
                for file in csv_files:
                    file_path = os.path.join(pzd_path, file)
                    # Read and process CSV in chunks
                    for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                        # Add pzd identifier
                        chunk['pzd_id'] = pzd_dir
                        
                        # Process chunk
                        processed_chunk = self.load_and_process_chunk(chunk)
                        processed_chunks.append(processed_chunk)
                        
                        # If we have enough chunks, combine them
                        if len(processed_chunks) * self.chunk_size >= self.chunk_size * 5:
                            combined_chunk = pd.concat(processed_chunks, ignore_index=True)
                            processed_chunks = [combined_chunk]
                            
                            # Force garbage collection
                            gc.collect()
            
        # Combine remaining chunks
        if not processed_chunks:
            raise ValueError("No CSV files found in data directories")
            
        combined_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Force garbage collection
        del processed_chunks
        gc.collect()
        
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for model training
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: X (features), y (target), feature_names
        """
        # Define feature columns
        feature_columns = [
            'days_to_otp',
            'pzd_kolmst_К', 'pzd_count_vag_К', 'kolsvm_К',
            'pzd_kolmst_Л', 'pzd_count_vag_Л', 'kolsvm_Л',
            'pzd_kolmst_М', 'pzd_count_vag_М', 'kolsvm_М',
            'pzd_kolmst_О', 'pzd_count_vag_О', 'kolsvm_О',
            'pzd_kolmst_П', 'pzd_count_vag_П', 'kolsvm_П',
            'pzd_kolmst_С', 'pzd_count_vag_С', 'kolsvm_С',
            'cum_count_tickets_by_pzd',
            'cum_count_tickets_lgot_255_by_pzd',
            'cum_count_tickets_lgot_by_pzd',
            'dow_dateotp', 'season',
            'is_holyday', 'is_pre_holyday',
            'cat_pre_holyday', 'school_holyday'
        ]
        
        # Define target columns
        target_columns = [
            'target_count_tickets_lgot_255',
            'target_count_tickets_lgot'
        ]
        
        # Prepare features in chunks
        X_chunks = []
        y_chunks = []
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            
            # Prepare features
            X_chunk = chunk[feature_columns].copy()
            y_chunk = chunk[target_columns].copy()
            
            # Scale numerical features
            numerical_columns = X_chunk.select_dtypes(include=['float64', 'int64']).columns
            X_chunk[numerical_columns] = self.scaler.fit_transform(X_chunk[numerical_columns])
            
            # Encode categorical features
            categorical_columns = X_chunk.select_dtypes(include=['category']).columns
            if len(categorical_columns) > 0:
                encoded_cats = self.encoder.fit_transform(X_chunk[categorical_columns])
                encoded_cats_df = pd.DataFrame(
                    encoded_cats,
                    columns=self.encoder.get_feature_names_out(categorical_columns)
                )
                X_chunk = pd.concat([X_chunk.drop(categorical_columns, axis=1), encoded_cats_df], axis=1)
            
            X_chunks.append(X_chunk.values)
            y_chunks.append(y_chunk.values)
            
            # Force garbage collection
            gc.collect()
        
        # Combine chunks
        X = np.vstack(X_chunks)
        y = np.vstack(y_chunks)
        
        # Clean up
        del X_chunks, y_chunks
        gc.collect()
        
        return X, y, X_chunk.columns.tolist()
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=42) 