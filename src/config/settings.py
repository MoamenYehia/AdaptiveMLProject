"""
Configuration and settings for Adaptive Fraud Detection System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data paths
DATA_PATH = DATA_DIR / "Synthetic_Financial_datasets_log.csv"

# Model parameters
MODEL_CONFIG = {
    "loss": "log_loss",  # for probability output
    "random_state": 42,
    "warm_start": True,
    "max_iter": 1000,
    "tol": 1e-3,
    "n_jobs": -1,
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "batch_size": 10000,
}

# Feature engineering
FEATURES_TO_DROP = ["nameOrig", "nameDest"]
CATEGORICAL_FEATURES = ["type"]
TYPE_CATEGORIES = ["CASH_IN", "PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"]

# Streamlit config
APP_CONFIG = {
    "page_title": "Adaptive Fraud Detection",
    "layout": "wide",
    "initial_sidebar_state": "auto",
}

# Project metadata
PROJECT_INFO = {
    "title": "🔍 Adaptive Financial Fraud Detection",
    "subtitle": "Online Learning Model that Adapts with New Data in Real-Time",
    "project_name": "Adaptive Machine Learning Model for Fraud Detection",
    "team": "Moamen yehia| Youssef Ekrami | Dareen Mohammed | Jomana Farag ",
    "dataset": "PaySim Synthetic Financial Dataset (Kaggle)",
}
