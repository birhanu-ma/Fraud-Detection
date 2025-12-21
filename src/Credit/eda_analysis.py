import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ImportantFeaturesEDA:
    def __init__(self, df: pd.DataFrame, target: str = 'Class'):
        self.df = df.copy()
        self.target = target
        # Create a readable label for plotting
        self.df['label'] = self.df[target].map({0: "Not Fraud", 1: "Fraud"})

    # 1. CHECK IMBALANCE (The missing method)
    def check_imbalance(self):
        print("=== Class Imbalance Quantification ===")
        counts = self.df[self.target].value_counts()
        percents = self.df[self.target].value_counts(normalize=True) * 100
        print(pd.DataFrame({'Count': counts, 'Percentage (%)': percents}))

        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x='label', palette='viridis')
        plt.title("Fraud vs. Not Fraud Distribution")
        plt.show()

    # 2. REMOVE OUTLIERS
    def remove_outliers(self, features, threshold=1.5):
        print(f"=== Removing Outliers for: {features} ===")
        initial_shape = self.df.shape[0]
        
        for col in features:
            fraud_data = self.df[self.df[self.target] == 1][col]
            q25, q75 = fraud_data.quantile(0.25), fraud_data.quantile(0.75)
            iqr = q75 - q25
            
            cut_off = iqr * threshold
            lower, upper = q25 - cut_off, q75 + cut_off
            
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            self.df = self.df.drop(outliers.index, errors='ignore')
            
            print(f"Feature {col}: Dropped {len(outliers)} rows.")

        print(f"Total rows removed: {initial_shape - self.df.shape[0]}")
        self.df['label'] = self.df[self.target].map({0: "Not Fraud", 1: "Fraud"})
        return self.df

    # 3. PLOT TIME & AMOUNT
    def plot_time_amount(self):
        print("=== Distribution of Time and Amount ===")
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data=self.df, x='Time', hue='label', kde=True, ax=ax[0], element="step")
        sns.histplot(data=self.df, x='Amount', hue='label', kde=True, ax=ax[1], element="step")
        ax[1].set_yscale('log')
        plt.show()

    # 4. BIVARIATE V-FEATURES
    def bivariate_v_features(self, features_to_plot=['V11', 'V12', 'V14', 'V17', 'V10', 'V16']):
        num_features = len(features_to_plot)
        rows = (num_features // 3) + (num_features % 3 > 0)
        fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(features_to_plot):
            sns.boxplot(x='label', y=col, data=self.df, ax=axes[i], palette='Set2')
        plt.tight_layout()
        plt.show()

    # 5. CORRELATION MATRIX
    def correlation_matrix(self):
        print("=== Feature Correlation Matrix (Top Correlated Features) ===")
        
        # Calculate correlation
        # We drop 'label' because it's a string; we need the numeric 'Class'
        corr = self.df.drop(columns=['label']).corr()
        
        # To make it readable, let's focus on how everything correlates with 'Class'
        # and only show the top 10 most influential features
        top_corr_cols = corr.nlargest(10, self.target)[self.target].index.tolist()
        top_corr_cols += corr.nsmallest(5, self.target)[self.target].index.tolist()
        top_corr_cols = list(set(top_corr_cols)) # Remove duplicates
        
        subset_corr = self.df[top_corr_cols].corr()

        plt.figure(figsize=(15, 10))
        
        # annot=True: Adds the numbers
        # fmt=".2f": Rounds to 2 decimal places
        # cmap='coolwarm': Blue for negative, Red for positive
        # center=0: Ensures the color neutral point is 0
        sns.heatmap(subset_corr, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='coolwarm', 
                    center=0,
                    linewidths=0.5,
                    cbar_kws={"shrink": .8})
        
        plt.title("Correlation Heatmap with Numeric Values")
        plt.show()

    # 6. RUN ALL
    def run_all(self, clean_outliers=False):
        # Now self.check_imbalance() will find the method defined above
        self.check_imbalance()
        
        if clean_outliers:
            self.remove_outliers(['V14', 'V12', 'V10'], threshold=1.5)
            
        self.plot_time_amount()
        self.bivariate_v_features(['V11', 'V12', 'V14', 'V17', 'V10', 'V16'])
        self.correlation_matrix()
        return self.df