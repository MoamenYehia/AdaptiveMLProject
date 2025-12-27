"""
Adaptive Fraud Detection Model - SGDClassifier with Online Learning

Features:
- Transfer learning initialization (via transfer_learning module)
- Immediate online updates on user feedback
- Class-weighted incremental training
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from typing import Optional, Callable, Dict
from .transfer_learning import initialize_with_transfer_learning

class AdaptiveFraudDetector:
    """
    Adaptive fraud detection model using SGDClassifier for online learning.
    Handles class imbalance with weighted samples.
    """
    
    def __init__(self, model_config):
        """
        Initialize the fraud detector model.
        
        Args:
            model_config (dict): Configuration parameters for SGDClassifier
        """
        self.model = SGDClassifier(**model_config)
        self.is_trained = False
        self.classes_ = np.array([0, 1])
    
    def train_initial(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      class_weight_dict: Dict[int, float],
                      batch_size: int = 10000,
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      enable_transfer_init: bool = True,
                      offline_init_sample_size: Optional[int] = None):
        """
        Train the model incrementally in batches for handling large datasets.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            class_weight_dict (dict): Class weights for imbalance handling
            batch_size (int): Batch size for incremental learning
            progress_callback (callable): Optional callback for progress updates
            enable_transfer_init (bool): If True, perform a short offline
                LogisticRegression fit on a sample to seed SGD coefficients.
            offline_init_sample_size (int|None): Number of samples for the
                offline initialization. Defaults to min(150_000, len(X_train)).
        """
        # Optional transfer-learning initialization
        if enable_transfer_init and len(X_train) > 0:
            coef, intercept = initialize_with_transfer_learning(
                X_train, y_train, class_weight_dict,
                sample_size=offline_init_sample_size,
                random_state=42,
            )
            if coef is not None:
                # Initialize SGD shapes via a tiny partial_fit, then override coef_
                self.model.partial_fit(X_train.iloc[:1], y_train.iloc[:1], classes=self.classes_)
                self.model.coef_ = coef
                self.model.intercept_ = intercept
                self.initialized_with_transfer = True

        num_batches = (len(X_train) + batch_size - 1) // batch_size
        
        for i in range(0, len(X_train), batch_size):
            end = min(i + batch_size, len(X_train))
            X_batch = X_train.iloc[i:end]
            y_batch = y_train.iloc[i:end]
            
            # Compute sample weights for this batch
            sample_weight_batch = np.where(
                y_batch == 0,
                class_weight_dict[0],
                class_weight_dict[1]
            )
            
            if i == 0 and not self.initialized_with_transfer:
                # First batch: provide classes (if not already initialized)
                self.model.partial_fit(
                    X_batch, y_batch,
                    classes=self.classes_,
                    sample_weight=sample_weight_batch
                )
            else:
                self.model.partial_fit(
                    X_batch, y_batch,
                    sample_weight=sample_weight_batch
                )
            
            if progress_callback:
                batch_num = (i // batch_size) + 1
                progress_callback(batch_num, num_batches)
        
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Probability estimates
        """
        return self.model.predict_proba(X)
    
    def adapt(self, X_new: pd.DataFrame, y_new: np.ndarray, class_weight_dict: Dict[int, float]):
        """
        Update model with new labeled data (online learning).
        
        Args:
            X_new (pd.DataFrame): New features
            y_new (np.ndarray): True labels for new data
            class_weight_dict (dict): Class weights
        """
        # Compute sample weights for incoming labels
        sample_weight = np.where(y_new == 0, class_weight_dict[0], class_weight_dict[1])

        # Ensure classes known (in case adapt is called before training)
        if not self.is_trained and not hasattr(self.model, "classes_"):
            self.model.partial_fit(X_new.iloc[:1], y_new[:1], classes=self.classes_)

        # Immediate online update so a single feedback changes metrics
        self.model.partial_fit(X_new, y_new, sample_weight=sample_weight)
        self.is_trained = True
    
    def get_model(self):
        """
        Get the underlying sklearn model.
        
        Returns:
            SGDClassifier: The trained model
        """
        return self.model

    # Optional persistence helpers (no UI changes required)
    def save(self, path: str):
        try:
            import joblib
            joblib.dump(self.model, path)
        except Exception:
            pass

    def load(self, path: str):
        try:
            import joblib
            self.model = joblib.load(path)
            self.is_trained = True
        except Exception:
            pass
