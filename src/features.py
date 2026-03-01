import pandas as pd

def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, drop_first=True)
