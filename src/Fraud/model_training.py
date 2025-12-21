import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    confusion_matrix
)

class FraudModelTrainer:
    def __init__(self, df, target_col="class", test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.results = {}

    def prepare_data(self):
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state
        )

    def train_logistic_regression(self):
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model

    def train_random_forest(self, n_estimators=300, max_depth=12):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model

    def evaluate_model(self, model_name):
        model = self.models[model_name]

        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            "AUC_PR": average_precision_score(self.y_test, y_proba),
            "F1_Score": f1_score(self.y_test, y_pred),
            "Confusion_Matrix": confusion_matrix(self.y_test, y_pred)
        }

        self.results[model_name] = metrics
        return metrics

    def cross_validate_model(self, model_name, k=5):
        model = self.models[model_name]

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        scoring = {
            "auc_pr": "average_precision",
            "f1": "f1"
        }

        cv_results = cross_validate(
            model,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        return {
            "AUC_PR_Mean": cv_results["test_auc_pr"].mean(),
            "AUC_PR_Std": cv_results["test_auc_pr"].std(),
            "F1_Mean": cv_results["test_f1"].mean(),
            "F1_Std": cv_results["test_f1"].std()
        }

    def compare_models(self):
        comparison = []

        for model_name, metrics in self.results.items():
            comparison.append({
                "Model": model_name,
                "AUC_PR": metrics["AUC_PR"],
                "F1_Score": metrics["F1_Score"]
            })

        return pd.DataFrame(comparison)
