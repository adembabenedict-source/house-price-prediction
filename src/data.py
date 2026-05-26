import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_sample_dataset():
    """Load a built-in sample housing dataset from scikit-learn."""
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    df['target'] = dataset.target
    return df


def load_data_from_csv(path: str, target_column: str = 'target') -> pd.DataFrame:
    """Load a CSV file containing house features and a target value."""
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")
    return df


def prepare_features(df: pd.DataFrame, target_column: str = 'target'):
    """Split DataFrame into features and target arrays."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
