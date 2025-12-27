# Adaptive Financial Fraud Detection System

## 🎯 Project Overview

An advanced machine learning system for **real-time fraud detection** using adaptive online learning. The model continuously improves by learning from new labeled transactions, making it ideal for detecting evolving fraud patterns.

**Dataset:** PaySim Synthetic Financial Dataset (Kaggle)

---

## 📁 Project Structure

```
project2/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main Streamlit application
│   ├── models/
│   │   ├── __init__.py
│   │   └── fraud_detector.py   # AdaptiveFraudDetector class
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Data loading & evaluation utilities
│   └── config/
│       ├── __init__.py
│       └── settings.py         # Configuration & constants
├── data/
│   └── Synthetic_Financial_datasets_log.csv
├── logs/                       # Application logs directory
├── models/                     # Trained models directory
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── run.sh / run.bat           # Quick start scripts
```

---

## ✨ Key Features

- **Online Learning:** Model adapts and improves with new labeled data in real-time
- **Balanced Training:** Handles class imbalance using weighted samples
- **Incremental Learning:** Uses SGDClassifier for memory-efficient batch training
- **Interactive UI:** Beautiful Streamlit interface for predictions and model updates
- **Performance Metrics:** Real-time accuracy, recall, and ROC-AUC tracking
- **Professional Architecture:** Modular, scalable, and maintainable code structure

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository** (or extract the project folder)
   ```bash
   cd path/to/project2
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Using Python directly:**
```bash
streamlit run src/main.py
```

**Using the batch script (Windows):**
```bash
run.bat
```

**Using the shell script (macOS/Linux):**
```bash
bash run.sh
```

The application will open in your default browser at `http://localhost:8501`

---

## 📊 How to Use

### 1. **View Current Performance**
   - Check the initial model accuracy, recall, and ROC-AUC scores
   - These metrics are computed on the test set

### 2. **Make Predictions**
   - Fill in the transaction details:
     - Transaction amount
     - Origin and destination account balances
     - Transaction type (CASH_IN, PAYMENT, CASH_OUT, TRANSFER, DEBIT)
     - Whether it's flagged for fraud
   - Click **"Predict Fraud on New Transaction"**
   - Get the prediction and fraud probability

### 3. **Adapt Model**
   - After making a prediction, verify it with the true label
   - Select the actual fraud status of the transaction
   - Click **"Update Model with This New Labeled Data"**
   - The model retrains and metrics are updated immediately

### 4. **Track Improvements**
   - Compare performance metrics before and after each update
   - Confusion matrices show how predictions changed
   - Continue adding labeled transactions to improve detection

---

## 🏗️ Architecture

### Core Components

#### **Models (src/models/fraud_detector.py)**
- `AdaptiveFraudDetector`: Main model wrapper
  - `train_initial()`: Batch training on initial dataset
  - `predict()`: Make predictions
  - `predict_proba()`: Get confidence scores
  - `adapt()`: Online learning with new data

#### **Utilities (src/utils/data_loader.py)**
- `load_data()`: Load CSV dataset
- `prepare_features()`: Feature engineering & encoding
- `compute_class_weights()`: Balance class imbalance
- `evaluate_model()`: Compute metrics
- `get_sample_weights()`: Calculate weights for samples

#### **Configuration (src/config/settings.py)**
- Centralized settings for:
  - Model hyperparameters
  - Training configuration
  - Feature engineering rules
  - Project metadata

---

## 🔧 Configuration

Edit `src/config/settings.py` to customize:

```python
# Model parameters
MODEL_CONFIG = {
    "loss": "log_loss",
    "random_state": 42,
    "warm_start": True,
    "max_iter": 1000,
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "batch_size": 10000,
}
```

---

## 📈 Model Details

**Algorithm:** Stochastic Gradient Descent (SGD) Classifier
- **Loss Function:** Log Loss (logistic regression)
- **Class Weighting:** Balanced (inverse of class frequencies)
- **Warm Start:** Enabled for incremental learning
- **Online Learning:** Uses `partial_fit()` for continuous adaptation

**Why SGDClassifier?**
- Memory-efficient for large datasets
- Supports online/incremental learning
- Handles sparse feature matrices
- Computationally efficient

---

## 📊 Metrics Explained

- **Accuracy:** Overall percentage of correct predictions
- **Recall (Fraud):** Percentage of actual frauds detected (minimize false negatives)
- **ROC AUC:** Area under the Receiver Operating Characteristic curve (0.5-1.0)

---

## 🐛 Troubleshooting

### Issue: "Dataset not found"
- Ensure `Synthetic_Financial_datasets_log.csv` exists in the `data/` directory
- Check the path in `src/config/settings.py`

### Issue: Port 8501 already in use
```bash
streamlit run src/main.py --server.port 8502
```

### Issue: Module not found errors
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## 🔐 Data Privacy

This project uses synthetic financial data generated by PaySim. No real financial information is used.

---

## 📝 Future Improvements

- [ ] Model persistence (save/load trained models)
- [ ] Batch prediction from CSV
- [ ] Advanced feature engineering
- [ ] Multiple model comparison
- [ ] Data drift detection
- [ ] API endpoint for integration
- [ ] Database integration for transaction history

---

## 📚 References

- [Scikit-learn SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## 📄 License

This project is part of the GSB-TRAINING 2025 program in collaboration with DIBIMBING.ID.

---

## 💬 Questions or Issues?

Feel free to reach out to the project team for support.

**Happy Fraud Detection! 🔍**
