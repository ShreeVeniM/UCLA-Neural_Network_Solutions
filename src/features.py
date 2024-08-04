import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import setup_logging

def engineer_features(df):
    try:
        # Example of feature engineering, you can add your own
        df['GRE_Score_Squared'] = df['GRE_Score'] ** 2
        return df
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return df

def scale_features(df, numeric_columns):
    try:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        return df
    except Exception as e:
        print(f"Error in scaling features: {e}")
        return df

def create_interaction_terms(df, interaction_pairs):
    try:
        for (col1, col2) in interaction_pairs:
            interaction_term = col1 + '_x_' + col2
            df[interaction_term] = df[col1] * df[col2]
        return df
    except Exception as e:
        print(f"Error in creating interaction terms: {e}")
        return df
