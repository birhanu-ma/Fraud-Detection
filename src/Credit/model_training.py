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
        # Efficient column dropping
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state
        )

    def train_logistic_regression(self):
        """Fastest training: Linear boundary."""
        # solver='liblinear' is often faster for small-medium datasets
        model = LogisticRegression(max_iter=500, solver='liblinear', random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model

    def train_random_forest(self, n_estimators=100):
        """Optimized Random Forest for speed."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, # Reduced from 300 to 100 for speed
            max_depth=10,              # Limited depth prevents massive trees
            n_jobs=-1,                 # USES ALL CPU CORES
            random_state=self.random_state,
            verbose=0
        )
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model

    def evaluate_model(self, model_name):
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        self.results[model_name] = {
            "AUC_PR": average_precision_score(self.y_test, y_proba),
            "F1_Score": f1_score(self.y_test, y_pred),
            "CM": confusion_matrix(self.y_test, y_pred)
        }
        
        # Quick Text Summary
        print(f"\n✅ {model_name} Complete.")
        print(f"F1: {self.results[model_name]['F1_Score']:.4f} | AUC-PR: {self.results[model_name]['AUC_PR']:.4f}")

    def plot_results(self):
        """Fast visualization of all trained models."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(12, 4))
        if len(self.results) == 1: axes = [axes] # Handle single model case
        
        for i, (name, metrics) in enumerate(self.results.items()):
            sns.heatmap(metrics["CM"], annot=True, fmt='d', cmap='Greens', ax=axes[i], cbar=False)
            axes[i].set_title(f"{name}\nF1: {metrics['F1_Score']:.2f}")
        plt.tight_layout()
        plt.show()