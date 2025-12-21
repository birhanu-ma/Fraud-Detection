import pytest
import pandas as pd
import numpy as np
from src.Fraud.model_training import FraudModelTrainer

# 1. Use a Fixture to create "Synthetic" data (Adoptability)
# This means the tests don't need your 100MB CSV file to run.
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'class': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    })

# 2. Test the "Happy Path" (Does it work?)
def test_prepare_data_success(sample_data):
    trainer = FraudModelTrainer(sample_data)
    trainer.prepare_data()
    assert trainer.X_train is not None
    assert trainer.y_train.shape[0] > 0

# 3. Defensive Test: Missing Column (Robustness)
def test_prepare_data_missing_column():
    # Data is missing the 'class' or 'Class' column
    df = pd.DataFrame({'wrong_col': [1, 2], 'value': [10, 20]})
    trainer = FraudModelTrainer(df)
    
    # We expect the code to raise a ValueError because it can't find the target
    with pytest.raises(ValueError, match="Target column not found"):
        trainer.prepare_data()

# 4. Defensive Test: Empty DataFrame
def test_empty_dataframe():
    df = pd.DataFrame()
    trainer = FraudModelTrainer(df)
    with pytest.raises(ValueError):
        trainer.prepare_data()

# 5. Defensive Test: All One Class (Handling extreme edge cases)
def test_single_class_error():
    # If data has ONLY legitimate transactions, SMOTE/Training should fail gracefully
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'class': [0, 0, 0, 0]
    })
    trainer = FraudModelTrainer(df)
    trainer.prepare_data()
    # Check if trainer handles the lack of minority class correctly
    with pytest.raises(Exception): 
        trainer.train_random_forest()