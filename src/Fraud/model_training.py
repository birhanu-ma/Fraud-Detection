import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Essential imports for model building and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
        """Prepares X and y, handling case-sensitivity for the target column."""
        # Check for 'Class' vs 'class' to avoid KeyError
        if self.target_col not in self.df.columns:
            if 'class' in self.df.columns:
                self.target_col = 'class'
            elif 'Class' in self.df.columns:
                self.target_col = 'Class'
            else:
                raise KeyError(f"Target column not found. Available: {self.df.columns.tolist()}")

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

    def train_logistic_regression(self):
        """Baseline model training."""
        model = LogisticRegression(max_iter=500, solver='liblinear', random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model
        print("✅ Logistic Regression trained.")

    def train_random_forest(self, n_estimators=100):
        """Optimized Ensemble model training."""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model
        print("✅ Random Forest trained.")

    def evaluate_model(self, model_name):
        """Calculates core metrics and stores them in self.results."""
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' has not been trained yet.")
            return

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        self.results[model_name] = {
            "AUC_PR": average_precision_score(self.y_test, y_proba),
            "F1_Score": f1_score(self.y_test, y_pred),
            "CM": confusion_matrix(self.y_test, y_pred)
        }
        print(f"✅ {model_name} Evaluated.")

    def cross_validate_model(self, model_name, k=5):
        """Performs Stratified K-Fold Cross Validation (Task 2 Requirement)."""
        print(f"Running {k}-fold Cross Validation for {model_name}...")
        model = self.models[model_name]

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        scoring = {
            "auc_pr": "average_precision",
            "f1": "f1"
        }

        cv_results = cross_validate(
            model, self.X, self.y,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        return {
            "Model": model_name,
            "AUC_PR_Mean": cv_results["test_auc_pr"].mean(),
            "AUC_PR_Std": cv_results["test_auc_pr"].std(),
            "F1_Mean": cv_results["test_f1"].mean(),
            "F1_Std": cv_results["test_f1"].std()
        }

    def compare_models(self):
        """Returns a side-by-side DataFrame comparison of all evaluated models."""
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                "Model": model_name,
                "AUC_PR": metrics["AUC_PR"],
                "F1_Score": metrics["F1_Score"]
            })
        return pd.DataFrame(comparison)

    def plot_results(self):
        """Displays Confusion Matrices for all evaluated models in the notebook."""
        if not self.results:
            print("❌ No evaluation results found. Run evaluate_model() first.")
            return

        fig, axes = plt.subplots(1, len(self.results), figsize=(12, 4))
        # Ensure axes is iterable even if there is only one model
        if len(self.results) == 1: axes = [axes]
        
        for i, (name, metrics) in enumerate(self.results.items()):
            sns.heatmap(metrics["CM"], annot=True, fmt='d', cmap='Greens', ax=axes[i], cbar=False)
            axes[i].set_title(f"{name}\nF1: {metrics['F1_Score']:.2f}")
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
            
        plt.tight_layout()
        plt.show()