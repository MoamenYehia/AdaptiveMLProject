"""
Utility functions for data loading and model evaluation
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def load_data(data_path):
    """
    Load the financial dataset from CSV.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def prepare_features(df, features_to_drop, categorical_cols):
    """
    Prepare features and target variable.
    
    Args:
        df (pd.DataFrame): Raw dataset
        features_to_drop (list): Columns to drop
        categorical_cols (list): Categorical columns for encoding
        
    Returns:
        tuple: (X, y) features and target
    """
    df_fe = df.drop(columns=features_to_drop, errors='ignore')
    X = df_fe.drop(columns=['isFraud'])
    y = df_fe['isFraud']
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, prefix='type')
    
    return X, y


def compute_class_weights(y_train):
    """
    Compute balanced class weights for imbalanced data.
    
    Args:
        y_train (pd.Series): Training labels
        
    Returns:
        dict: Class weight dictionary
    """
    classes = np.array([0, 1])
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights_array))
    
    return class_weight_dict


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained classifier
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        
    Returns:
        tuple: (accuracy, recall, auc, predictions, probabilities, confusion_matrix)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, recall, auc, y_pred, y_proba, cm


def get_sample_weights(y_batch, class_weight_dict):
    """
    Get sample weights for a batch of data.
    
    Args:
        y_batch (pd.Series): Batch labels
        class_weight_dict (dict): Class weight dictionary
        
    Returns:
        np.ndarray: Sample weights
    """
    return np.where(
        y_batch == 0,
        class_weight_dict[0],
        class_weight_dict[1]
    )
