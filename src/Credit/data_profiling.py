import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

class DataProfiling:
    """
    Performs data profiling specifically optimized for 
    numerical PCA features (V1-V28), Time, Amount, and Class.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ============================================================
    # 1️⃣ DATA OVERVIEW
    # ============================================================
    def overview(self):
        print("====== DATA OVERVIEW ======")
        print(f"Total Transactions: {self.df.shape[0]}")
        print(f"Total Features: {self.df.shape[1]}")
        print("\nData Types Summary:")
        print(self.df.dtypes.value_counts())
        print("\nFirst 3 Rows:")
        display(self.df.head(3))

    # ============================================================
    # 2️⃣ SUMMARY STATISTICS
    # ============================================================
    def summary_statistics(self):
        print("\n====== SUMMARY STATISTICS ======")

        # Grouping columns by type
        # Time and Amount are usually the most relevant 'raw' numeric features
        main_numeric = ['Time', 'Amount']
        pca_features = [col for col in self.df.columns if col.startswith('V')]
        
        print("\n--- Main Features (Time/Amount) ---")
        display(self.df[main_numeric].describe().transpose())

        print("\n--- PCA Features (V1-V28) Statistics ---")
        # We only show a summary of V-features to avoid cluttering the screen
        pca_stats = self.df[pca_features].describe().transpose()
        display(pca_stats.head(5)) # Showing first 5 for brevity
        print(f"... and {len(pca_features)-5} more PCA features.")

        # Class Label Analysis
        if 'Class' in self.df.columns:
            print("\n--- Target Variable (Class) Distribution ---")
            counts = self.df['Class'].value_counts()
            percents = self.df['Class'].value_counts(normalize=True) * 100
            dist_df = pd.DataFrame({'Count': counts, 'Percentage %': percents})
            display(dist_df)
            print("> Note: 0 = Legit, 1 = Fraud")

    # ============================================================
    # 3️⃣ MISSING & DUPLICATE VALUES
    # ============================================================
    def data_quality(self):
        print("\n====== DATA QUALITY CHECK ======")
        
        # Missing Values
        missing = self.df.isnull().sum().sum()
        print(f"Total Missing Values: {missing}")
        
        # Duplicate Rows (Common in this dataset)
        dup_count = self.df.duplicated().sum()
        print(f"Duplicate Rows Found: {dup_count}")
        if dup_count > 0:
            print(f"Percentage of Duplicates: {(dup_count/len(self.df)*100):.2f}%")

    # ============================================================
    # 4️⃣ CORRELATION SNAPSHOT
    # ============================================================
    def correlation_check(self):
        """Quick check of feature correlation with the Class variable."""
        if 'Class' in self.df.columns:
            print("\n====== TOP CORRELATIONS WITH CLASS ======")
            corr = self.df.corr()['Class'].sort_values(ascending=False)
            print("Top Positive Correlations:\n", corr.head(4))
            print("\nTop Negative Correlations:\n", corr.tail(3))

    # ============================================================
    # 5️⃣ RUN ALL
    # ============================================================
    def run_all(self):
        """
        Runs the profiling pipeline for the Credit Card dataset.
        """
        self.overview()
        self.summary_statistics()
        self.data_quality()
        self.correlation_check()

        return self.df