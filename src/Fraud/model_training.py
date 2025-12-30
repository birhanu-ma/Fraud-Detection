import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
        """
        Prepares X and y with defensive checks for data integrity.
        Updates: Added check for empty DF and switched KeyError to ValueError.
        """
        # 1. Defensive Check: Empty DataFrame
        if self.df.empty:
            raise ValueError("Cannot prepare data: The provided DataFrame is empty.")

        # 2. Defensive Check: Target column identification
        if self.target_col not in self.df.columns:
            if 'class' in self.df.columns:
                self.target_col = 'class'
            elif 'Class' in self.df.columns:
                self.target_col = 'Class'
            else:
                # Switched to ValueError to satisfy robust testing requirements
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

    def _check_class_diversity(self):
        """Internal helper to ensure both classes are present before training."""
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("Training aborted: The training set contains only one class.")

    def train_logistic_regression(self):
        """Baseline model training with defensive class check."""
        self._check_class_diversity()
        
        model = LogisticRegression(max_iter=500, solver='liblinear', random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model
        print("✅ Logistic Regression trained.")

    def train_random_forest(self, n_estimators=100):
        """Optimized Ensemble model training with defensive class check."""
        self._check_class_diversity()

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
        """Performs Stratified K-Fold Cross Validation."""
        if model_name not in self.models:
             raise ValueError(f"Model '{model_name}' must be trained before cross-validation.")

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
        """Returns a side-by-side DataFrame comparison."""
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({
                "Model": model_name,
                "AUC_PR": metrics["AUC_PR"],
                "F1_Score": metrics["F1_Score"]
            })
        return pd.DataFrame(comparison)
 
    def plot_results(self):
          """Fast visualization of all trained models."""
          if not self.results:
              print("No results to plot. Run evaluate_model() first.")
              return
  
          fig, axes = plt.subplots(1, len(self.results), figsize=(12, 4))
          
          # Handle the case where there is only one model (axes is not an array)
          if len(self.results) == 1: 
              axes = [axes] 
          
          for i, (name, metrics) in enumerate(self.results.items()):
              # Change "CM" to "Confusion_Matrix" to match evaluate_model
              sns.heatmap(metrics["Confusion_Matrix"], annot=True, fmt='d', cmap='Greens', ax=axes[i], cbar=False)
              axes[i].set_title(f"{name}\nF1: {metrics['F1_Score']:.2f}")
              axes[i].set_xlabel('Predicted')
              axes[i].set_ylabel('Actual')
              
          plt.tight_layout()
          plt.show()
    def get_feature_importance(self, model_name="Random Forest"):
        """Task 3: Extract importance with a single default color."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained.")
        
        model = self.models[model_name]
        importances = model.feature_importances_
        feature_names = self.X.columns
        
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
    
        plt.figure(figsize=(10, 6))
        # Use color='steelblue' for a single, consistent bar color
        sns.barplot(x='Importance', y='Feature', data=fi_df, color='steelblue')
        plt.title(f"Top 10 Feature Importances ({model_name})")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()

    def run_shap_analysis(self, model_name="Random Forest", n_samples=100):
        """Task 3: Generate SHAP Summary Plot with shape fix."""
        model = self.models[model_name]
        
        # Select the sample
        test_sample = self.X_test.head(n_samples)
        
        # Initialize the explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate values
        shap_values = explainer.shap_values(test_sample)
    
        print(f"Generating SHAP Summary Plot for {model_name}...")
        
        # --- THE FIX ---
        # Sometimes Random Forest shap_values is a list, sometimes a 3D array 
        # depending on the SHAP version. We ensure we grab the "Fraud" (class 1) values.
        if isinstance(shap_values, list):
            # If it's a list, we want index 1
            val_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            # If it's a 3D array (samples, features, classes), grab class 1
            val_to_plot = shap_values[:, :, 1]
        else:
            # Default fallback
            val_to_plot = shap_values
    
        # Ensure feature names match exactly
        shap.summary_plot(val_to_plot, test_sample, feature_names=test_sample.columns)
        
        return explainer, shap_values

    def plot_specific_prediction(self, explainer, case_type="TP"):
        """Task 3: Robust Force Plot for specific predictions."""
        # 1. Get predictions for the whole test set
        model = self.models["Random Forest"]
        y_pred = model.predict(self.X_test)
        y_true = self.y_test.values
        
        # 2. Find the index in X_test
        if case_type == "TP":
            indices = np.where((y_pred == 1) & (y_true == 1))[0]
            title = "True Positive (Correctly Flagged Fraud)"
        elif case_type == "FP":
            indices = np.where((y_pred == 1) & (y_true == 0))[0]
            title = "False Positive (Legitimate Flagged as Fraud)"
        elif case_type == "FN":
            indices = np.where((y_pred == 0) & (y_true == 1))[0]
            title = "False Negative (Missed Fraud)"
        
        if len(indices) == 0:
            print(f"No cases found for {case_type}")
            return
    
        idx = indices[0]
        
        # 3. Get data for that specific row and ensure it is a 2D array/DataFrame
        row_data = self.X_test.iloc[[idx]] # Double brackets [[ ]] keep it as a DataFrame (2D)
        
        # 4. Calculate SHAP values specifically for this row to avoid DimensionErrors
        # This ensures the SHAP matrix and the row_data matrix are identical in size
        row_shap_values = explainer.shap_values(row_data)
    
        # Handle the list/array structure for Random Forest
        if isinstance(row_shap_values, list):
            val_to_plot = row_shap_values[1][0] # Class 1, first row
        elif len(row_shap_values.shape) == 3:
            val_to_plot = row_shap_values[0, :, 1] # First row, all features, Class 1
        else:
            val_to_plot = row_shap_values[0]
    
        print(f"Plotting {title}...")
        
        # Use matplotlib=True to ensure it displays in the notebook cell
        return shap.force_plot(
            explainer.expected_value[1], 
            val_to_plot, 
            row_data.iloc[0], # Pass the row as a Series for the labels
            matplotlib=True
        )
