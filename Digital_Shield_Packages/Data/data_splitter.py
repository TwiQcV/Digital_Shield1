import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_col, test_size=0.3, random_state=42):
    """
    Split dataset into train and test sets

    Args:
        df: pandas DataFrame
        target_col: name of target column
        test_size: proportion for test set (default 0.2)
        random_state: random seed (default 42)

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
