import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ImportantFeaturesEDA:
    """
    EDA class for selected important features: country, age, sex, browser, source, purchase_time.
    Performs univariate, bivariate, and multivariate analysis with proper binning for age and purchase time.
    Country-related plots only show top 10 countries by transaction count.
    Includes class imbalance analysis.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.original_target = target

        # Map binary target to readable labels
        if set(self.df[target].dropna().unique()) <= {0, 1}:
            self.target = target + "_label"
            self.df[self.target] = self.df[target].map({0: "Not Fraud", 1: "Fraud"})
        else:
            self.target = target

        # Bin age into 10-year intervals
        if 'age' in self.df.columns:
            bins = list(range(0, 101, 10))
            labels = [f"{b+1}-{b+10}" for b in bins[:-1]]
            self.df['age_group'] = pd.cut(self.df['age'], bins=bins, labels=labels, right=True)

        # Bin purchase_time into 3-hour intervals
        if 'purchase_time' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['purchase_time']):
                self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
            self.df['purchase_hour'] = self.df['purchase_time'].dt.hour
            bins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
            labels = [f"{b}-{b+3}" for b in bins[:-1]]
            self.df['purchase_time_group'] = pd.cut(self.df['purchase_hour'], bins=bins, labels=labels, right=False)

    # ============================================================
    # Class Distribution
    # ============================================================
    def class_distribution(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.df, x=self.target)
        plt.title("Target Class Distribution")
        plt.show()

    # ============================================================
    # Class Imbalance Quantification
    # ============================================================
    def class_imbalance(self):
        """
        Prints counts and percentages of each class to quantify imbalance.
        """
        counts = self.df[self.original_target].value_counts()
        percentages = (counts / len(self.df)) * 100
        imbalance_df = pd.DataFrame({
            'Count': counts,
            'Percentage (%)': percentages
        })
        print("=== Class Imbalance ===")
        display(imbalance_df)

    # ============================================================
    # Bivariate Analysis (important features vs fraud)
    # ============================================================
    def bivariate_analysis(self):
        important_cols = ['country', 'age_group', 'sex', 'browser', 'source', 'purchase_time_group']
        for col in important_cols:
            if col not in self.df.columns:
                continue
            plt.figure(figsize=(8, 4))
            # For country, show top 10 only
            if col == 'country':
                top_countries = self.df['country'].value_counts().nlargest(10).index
                sns.countplot(data=self.df[self.df['country'].isin(top_countries)],
                              x=col, hue=self.target, order=top_countries)
            else:
                sns.countplot(data=self.df, x=col, hue=self.target,
                              order=self.df[col].value_counts().index)
            plt.title(f"{col} vs {self.target}")
            plt.xticks(rotation=45)
            plt.show()

    # ============================================================
    # Multivariate Analysis: Country vs Age for fraud=1
    # ============================================================
    def multivariate_country_age_fraud(self):
        if 'country' not in self.df.columns or 'age_group' not in self.df.columns:
            print("Columns 'country' or 'age_group' not found.")
            return

        fraud_df = self.df[self.df[self.original_target] == 1]
        top_countries = self.df['country'].value_counts().nlargest(10).index
        fraud_df = fraud_df[fraud_df['country'].isin(top_countries)]

        plt.figure(figsize=(12, 6))
        sns.countplot(data=fraud_df, x='country', hue='age_group',
                      order=top_countries)
        plt.title("Fraudulent Transactions by Country and Age Group (Top 10 Countries)")
        plt.xticks(rotation=45)
        plt.show()

    # ============================================================
    # Run all analyses
    # ============================================================
    def run_all(self):
        print("=== Class Distribution ===")
        self.class_distribution()
        self.class_imbalance()
        print("=== Bivariate Analysis ===")
        self.bivariate_analysis()
        print("=== Multivariate Analysis (Country vs Age for Fraud) ===")
        self.multivariate_country_age_fraud()
