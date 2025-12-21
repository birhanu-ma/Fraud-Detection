import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataProfiling:
    """
    Performs data profiling operations on the input dataframe,
    including IP-to-country mapping.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ============================================================
    # 1️⃣ IP ADDRESS → COUNTRY MAPPING
    # ============================================================
    def add_country_column(self, ip_country_path: str):
        """
        Adds a 'country' column by mapping ip_address to country
        using an IP range lookup file. Unmatched IPs remain NaN.
        """

        print("====== ADDING COUNTRY COLUMN ======")

        ip_country = pd.read_csv(ip_country_path)

        # Ensure numeric IPs
        self.df['ip_address'] = self.df['ip_address'].astype(np.int64)
        ip_country['lower_bound_ip_address'] = ip_country['lower_bound_ip_address'].astype(np.int64)
        ip_country['upper_bound_ip_address'] = ip_country['upper_bound_ip_address'].astype(np.int64)

        # Sort for efficient merge_asof
        self.df = self.df.sort_values('ip_address')
        ip_country = ip_country.sort_values('lower_bound_ip_address')

        # Merge-as-of to get closest lower bound
        self.df = pd.merge_asof(
            self.df,
            ip_country[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
            left_on='ip_address',
            right_on='lower_bound_ip_address',
            direction='backward'
        )

        # Keep only IPs within range, else set country to NaN
        self.df.loc[
            (self.df['ip_address'] < self.df['lower_bound_ip_address']) |
            (self.df['ip_address'] > self.df['upper_bound_ip_address']),
            'country'
        ] = np.nan

        # Cleanup
        self.df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

        print("Country column added successfully. Unmatched IPs are NaN.")

    # ============================================================
    # 2️⃣ DATA OVERVIEW
    # ============================================================
    def overview(self):
        print("====== DATA OVERVIEW ======")
        print(f"Shape: {self.df.shape}")
        print("\nData Types:\n", self.df.dtypes)
        print("\nFirst 5 Rows:\n", self.df.head())

    # ============================================================
    # 3️⃣ SUMMARY STATISTICS
    # ============================================================
    def summary_statistics(self):
        print("\n====== SUMMARY STATISTICS ======")

        # Numeric features
        numeric_cols = ['purchase_value', 'age']
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]

        if numeric_cols:
            print("\n--- Numeric Features ---")
            display(self.df[numeric_cols].describe().transpose())

        # Categorical features (exclude IDs)
        id_cols = ['user_id', 'device_id', 'ip_address']
        cat_cols = [
            col for col in self.df.columns
            if col not in numeric_cols + id_cols
            and self.df[col].dtype == 'object'
        ]

        if cat_cols:
            print("\n--- Categorical Features ---")
            for col in cat_cols:
                print(f"\nFeature: {col}")
                print(f"Unique values: {self.df[col].nunique()}")
                print(f"Top 5 most frequent values:\n{self.df[col].value_counts().head()}")

    # ============================================================
    # 4️⃣ MISSING VALUES
    # ============================================================
    def missing_values(self):
        print("\n====== MISSING VALUES ======")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        mv = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })

        display(mv.sort_values('Missing %', ascending=False))

    # ============================================================
    # 5️⃣ DUPLICATE ROWS
    # ============================================================
    def duplicate_rows(self):
        print("\n====== DUPLICATE ROWS ======")
        dup_count = self.df.duplicated().sum()
        print(f"Duplicate Rows: {dup_count}")

    # ============================================================
    # 6️⃣ UNIQUE VALUES PER COLUMN
    # ============================================================
    def column_uniques(self):
        print("\n====== UNIQUE VALUES PER COLUMN ======")
        display(self.df.nunique().sort_values())

    # ============================================================
    # 7️⃣ RUN ALL
    # ============================================================
    def run_all(self, ip_country_path: str = None):
        """
        Runs all profiling tasks.
        If ip_country_path is provided, country column is added first.
        Returns the processed DataFrame.
        """

        if ip_country_path and 'country' not in self.df.columns:
            self.add_country_column(ip_country_path)

        self.overview()
        self.summary_statistics()
        self.missing_values()
        self.duplicate_rows()
        self.column_uniques()  

        return self.df
