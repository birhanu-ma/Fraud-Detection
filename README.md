# Fraud-Detection
Fraud Detection for E-Commerce and Banking 🛡️
📌 Project Overview
Adey Innovations Inc. specializes in fintech solutions for e-commerce and banking. This project implements a sophisticated fraud detection system designed to identify fraudulent transactions across two distinct domains:

E-Commerce: Identifying bot-like behavior and registration-to-purchase velocity risks.

Banking: Detecting anomalies in credit card transactions using anonymized PCA features.

A core focus of this project is managing the Class Imbalance problem and utilizing Model Explainability (SHAP) to ensure that security measures do not compromise the user experience for legitimate customers.

📂 Project Structure
Plaintext

fraud-detection/
├── .github/workflows/      # CI/CD: Automated linting and unit tests
├── data/                   # (Gitignored) Raw and Processed datasets
├── models/                 # Serialized model artifacts (.pkl)
├── notebooks/              # Step-by-step experimentation
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb      # Task 2: Training & Evaluation
│   └── shap-explainability.ipynb
├── src/                    # Production-grade source code
│   ├── Fraud/              # E-commerce specific logic
│   └── Credit/             # Banking specific logic
├── tests/                  # Pytest unit tests
├── requirements.txt        # Project dependencies
└── README.md
🚀 Getting Started
1. Prerequisites
Python 3.9 or higher

pip or conda

2. Installation
Bash

# Clone the repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Usage
To train the models and generate evaluation metrics, you can run the notebooks in the notebooks/ directory or use the source scripts:

Python

from src.Fraud.model_training import FraudModelTrainer
import pandas as pd

df = pd.read_csv("data/processed/fraud_data.csv")
trainer = FraudModelTrainer(df)
trainer.prepare_data()
trainer.train_random_forest()
trainer.plot_results()

## ✅ Testing
To ensure the robustness of the fraud detection pipeline, we use `pytest`.
Our tests cover:
- **Data Validation:** Ensures required columns are present.
- **Resampling Accuracy:** Verifies SMOTE balances the classes correctly.
- **Model Output:** Confirms prediction shapes and data types.

Run tests with:
`pytest tests/`

### 🛠️ Defensive Engineering
This project implements defensive programming practices:
- **Input Validation:** The `FraudModelTrainer` validates the existence of target columns (`class`/`Class`) before execution.
- **Robust Testing:** The test suite in `tests/` includes edge-case testing (empty datasets, missing features) to ensure production stability.
- **CI Automation:** Every push is automatically tested via GitHub Actions to prevent regressions.