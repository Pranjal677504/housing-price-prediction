"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_housing_data(filepath=None):
    """
    Load housing dataset from a CSV file or use sample data if file doesn't exist.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Housing dataset
    """
    if filepath and os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # Generate sample housing data for demonstration
        print("Generating sample housing dataset...")
        np.random.seed(42)
        n_samples = 506
        
        data = {
            'CRIM': np.random.uniform(0.006, 88, n_samples),
            'ZN': np.random.uniform(0, 100, n_samples),
            'INDUS': np.random.uniform(0.46, 27.74, n_samples),
            'CHAS': np.random.randint(0, 2, n_samples),
            'NOX': np.random.uniform(0.385, 0.871, n_samples),
            'RM': np.random.uniform(3.561, 8.78, n_samples),
            'AGE': np.random.uniform(2.9, 100, n_samples),
            'DIS': np.random.uniform(1.13, 12.13, n_samples),
            'RAD': np.random.randint(1, 25, n_samples),
            'TAX': np.random.uniform(187, 711, n_samples),
            'PTRATIO': np.random.uniform(12.6, 22, n_samples),
            'B': np.random.uniform(0.32, 396.9, n_samples),
            'LSTAT': np.random.uniform(1.73, 37.97, n_samples),
        }
        
        # Generate target with some relationship to features
        df = pd.DataFrame(data)
        price = (
            df['RM'] * 8 +
            -df['LSTAT'] * 0.5 +
            df['PTRATIO'] * -1 +
            np.random.normal(0, 5, n_samples)
        ) * 1000
        df['PRICE'] = np.maximum(price, 5000)  # Ensure positive prices
        
        return df


def preprocess_data(df, target_column='PRICE'):
    """
    Preprocess data by handling missing values and scaling features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
        
    Returns:
        tuple: (features, target, feature_names)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def prepare_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with scaling.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target values
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler
