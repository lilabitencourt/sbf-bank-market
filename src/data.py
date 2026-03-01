import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def map_target(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    df[target_col] = df[target_col].map({"yes": 1, "no": 0})
    return df
