import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def apply_one_hot_encoding(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Apply One-Hot Encoding on categorical columns and return transformed train/test sets.

    Parameters:

    X_train (pd.DataFrame): Training features
    X_test (pd.DataFrame): Test features

    Returns:
    X_train (pd.DataFrame): Encoded training data
    X_test (pd.DataFrame): Encoded test data
    """

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(categorical_cols)}")

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit on training data and transform both train and test
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # Create encoded DataFrames
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_col_names, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_col_names, index=X_test.index)

    # Drop original categorical columns
    X_train = X_train.drop(columns=categorical_cols)
    X_test = X_test.drop(columns=categorical_cols)

    # Concatenate encoded columns with remaining data
    X_train_processed = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test_processed = pd.concat([X_test, X_test_encoded_df], axis=1)

    print("✅ One-Hot Encoding applied successfully on training and test data.")
    return X_train_processed, X_test_processed
