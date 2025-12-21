import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class FraudModelTrainer:
    def __init__(self, df, target_col="Class", test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def prepare_data(self):
        """Prepares X and y with defensive checks for data integrity."""
        # Defensive Check: Empty DataFrame
        if self.df.empty:
            raise ValueError("Target column not found. Available: []")

        # Defensive Check: Case-sensitivity and existence of target
        if self.target_col not in self.df.columns:
            if 'class' in self.df.columns:
                self.target_col = 'class'
            elif 'Class' in self.df.columns:
                self.target_col = 'Class'
            else:
                raise ValueError(f"Target column not found. Available: {self.df.columns.tolist()}")

        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        # Stratified split to preserve class distribution
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state
        )
        print(f"✅ Data prepared. Target column identified as: '{self.target_col}'")

    def _validate_training(self):
        """Internal defensive check to ensure model can be trained."""
        if not hasattr(self, 'y_train'):
            raise ValueError("Data not prepared. Call prepare_data() before training.")
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Cannot train model with only one class in the target variable.")

    def train_logistic_regression(self):
        """Fast training with defensive check."""
        self._validate_training()
        
        model = LogisticRegression(max_iter=500, solver='liblinear', random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model
        print("✅ Logistic Regression trained.")

    def train_random_forest(self, n_estimators=100):
        """Optimized for speed and robustness."""
        self._validate_training()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=-1,  # Uses all CPU cores
            random_state=self.random_state,
            verbose=0
        )
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model
        print("✅ Random Forest trained.")

    def evaluate_model(self, model_name):
        """Evaluates model and prints summary."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been trained yet.")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        self.results[model_name] = {
            "AUC_PR": average_precision_score(self.y_test, y_proba),
            "F1_Score": f1_score(self.y_test, y_pred),
            "CM": confusion_matrix(self.y_test, y_pred)
        }
        
        print(f"\n✅ {model_name} Complete.")
        print(f"F1: {self.results[model_name]['F1_Score']:.4f} | AUC-PR: {self.results[model_name]['AUC_PR']:.4f}")

    def plot_results(self):
        """Visualization of confusion matrices."""
        if not self.results:
            print("❌ No evaluation results found. Run evaluate_model() first.")
            return

        fig, axes = plt.subplots(1, len(self.results), figsize=(12, 4))
        if len(self.results) == 1: axes = [axes]
        
        for i, (name, metrics) in enumerate(self.results.items()):
            sns.heatmap(metrics["CM"], annot=True, fmt='d', cmap='Greens', ax=axes[i], cbar=False)
            axes[i].set_title(f"{name}\nF1: {metrics['F1_Score']:.2f}")
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
            
        plt.tight_layout()
        plt.show()