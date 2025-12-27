"""
Transfer Learning Module for Fraud Detection

Provides transfer-learning-style initialization for SGDClassifier by fitting
a LogisticRegression model on a stratified sample and seeding SGD coefficients.
This improves convergence and recall on minority classes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple, Optional


def initialize_with_transfer_learning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight_dict: Dict[int, float],
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a LogisticRegression model on a stratified sample to seed SGD coefficients.

    Args:
        X_train (pd.DataFrame): Full training features
        y_train (pd.Series): Full training labels
        class_weight_dict (dict): Class weights {0: weight_0, 1: weight_1}
        sample_size (int|None): Number of samples to use for LR fit.
                                Defaults to min(150_000, len(X_train)).
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (coef, intercept) from fitted LogisticRegression
    """
    if len(X_train) == 0:
        return None, None

    # Determine sample size
    n_samples = sample_size or min(150_000, len(X_train))
    n_samples = min(n_samples, len(X_train))

    # Stratified sample to preserve class distribution
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=n_samples, replace=False)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]

    # Fit LogisticRegression with class weights
    lr = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=200,
        solver="lbfgs",
        random_state=random_state,
    )
    lr.fit(X_sample, y_sample)

    return lr.coef_.copy(), lr.intercept_.copy()
