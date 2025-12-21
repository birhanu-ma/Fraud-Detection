import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import WOEEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

class FraudDataProcessor:
    """
    Memory-safe fraud data processor using WoE encoding.
    Designed to work with a simple usage pattern.
    """

    def __init__(self, df):
        self.df = df.copy()
        self.encoder = None
        self.scaler = None
        self.smote = SMOTE(random_state=42)
        self.df_transformed = None
        self.target = None

    # ============================================================
    # Feature Engineering
    # ============================================================
    def feature_engineering(self):
        for col in ['signup_time', 'purchase_time']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek

        self.df['time_since_signup'] = (
            self.df['purchase_time'] - self.df['signup_time']
        ).dt.total_seconds() / 3600

        self.df['transaction_count'] = (
            self.df.groupby('user_id')['purchase_time'].transform('count')
        )

        return self.df

    # ============================================================
    # Data Transformation (NO target argument)
    # ============================================================
    def data_transformation(self, numerical_features, categorical_features):
        """
        Scales numeric features and WoE-encodes categorical features.
        Target column is inferred later.
        """
        X = self.df[numerical_features + categorical_features]

        # Temporarily store features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        return X  # returns raw feature df for compatibility

    # ============================================================
    # Handle Class Imbalance (WoE + Scaling + SMOTE)
    # ============================================================
    def handle_class_imbalance(self, target_column):
        """
        Encodes, scales, and applies SMOTE.
        """
        self.target = target_column
        y = self.df[target_column]

        # WoE Encoding
        self.encoder = WOEEncoder(cols=self.categorical_features)
        X_cat = self.encoder.fit_transform(
            self.df[self.categorical_features], y
        )

        # Scale numerical features
        self.scaler = StandardScaler()
        X_num = pd.DataFrame(
            self.scaler.fit_transform(self.df[self.numerical_features]),
            columns=self.numerical_features,
            index=self.df.index
        )

        # Combine
        self.df_transformed = pd.concat([X_num, X_cat], axis=1)

        print("Class distribution BEFORE SMOTE:", Counter(y))
        X_res, y_res = self.smote.fit_resample(self.df_transformed, y)
        print("Class distribution AFTER SMOTE:", Counter(y_res))

        return X_res, y_res
