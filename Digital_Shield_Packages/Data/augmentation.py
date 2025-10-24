from imblearn.over_sampling import SMOTE
import pandas as pd

def augment_data(X_train_processed, y_train):
    """
    Apply SMOTE oversampling to balance the dataset and fix one-hot encoded columns.
    """
    # Get original class counts
    class_counts = y_train.value_counts()

    # Calculate desired counts
    sampling_strategy = {
    'Critical': 4369,  # 85% of majority
    'High': 5141,      # Keep as is
    'Low': 4369,       # Bring up
    'Medium': 4369     # Bring up
    }

    # Apply SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_encoded_aug, y_train_aug = smote.fit_resample(X_train_processed, y_train)

    # Convert back to DataFrame
    X_train_encoded_aug = pd.DataFrame(X_train_encoded_aug, columns=X_train_processed.columns)

    # FIX: Round one-hot encoded columns back to 0 or 1
    # Identify one-hot encoded columns (adjust patterns to match your column names)
    ohe_patterns = ['country', 'attack type', 'target industry', 'attack source',
                    'security vulnerability type', 'defense mechanism used_']

    ohe_cols = [col for col in X_train_encoded_aug.columns
                if any(pattern in col for pattern in ohe_patterns)]

    # Round to nearest integer (0 or 1)
    X_train_encoded_aug[ohe_cols] = X_train_encoded_aug[ohe_cols].round().astype(int)

    # Combine into a new balanced DataFrame
    balanced_df = pd.concat([X_train_encoded_aug, y_train_aug], axis=1)

    return X_train_encoded_aug, y_train_aug, balanced_df
