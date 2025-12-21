import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

class FraudDataProcessor:
    """
    Final stage processor: 
    1. Removes non-feature columns (label, Time).
    2. Standardizes 'Amount' specifically using RobustScaler.
    3. Returns a train-ready resampled dataset.
    """

    def __init__(self, df):
        # Ensure we don't modify the original dataframe outside the class
        self.df = df.copy()
        self.scaler = RobustScaler() 
        self.smote = SMOTE(random_state=42)

    # ============================================================
    # 1. Feature Engineering
    # ============================================================
    def feature_engineering(self):
        """
        Transforms raw seconds into Hour of Day and removes original Time.
        """
        print("=== Performing Feature Engineering ===")
        
        if 'Time' in self.df.columns:
            # Convert Time (seconds) to Hour of Day (0-23)
            self.df['hour_of_day'] = (self.df['Time'] // 3600) % 24
            # Remove raw Time as it is a non-standardized sequence counter
            self.df.drop('Time', axis=1, inplace=True)
            
        return self.df

    # ============================================================
    # 2. Scaling & Imbalance Handling (Train-Ready Output)
    # ============================================================
    def prepare_train_ready_data(self, target_column='Class'):
        """
        Removes label, standardizes Amount/Features, and applies SMOTE.
        Returns X_res, y_res ready for model training.
        """
        print("=== Standardizing and Preparing Train-Ready Data ===")
        
        # 1. Remove 'label' and isolate target
        # The 'label' column is purely for EDA; it must be removed for training
        y = self.df[target_column]
        X = self.df.drop(columns=[target_column, 'label'], errors='ignore')

        # 2. Standardize Features
        # We use RobustScaler because 'Amount' typically has extreme outliers 
        # that would distort StandardScaler.
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 3. Apply SMOTE to balance the minority class (Fraud)
        # Standardizing BEFORE SMOTE prevents synthetic noise creation.
        print("Class distribution BEFORE SMOTE:", Counter(y))
        
        X_res, y_res = self.smote.fit_resample(X_scaled, y)
        
        print("Class distribution AFTER SMOTE:", Counter(y_res))
        print("--- Dataset is now numeric and train-ready ---")

        return X_res, y_res
    

    