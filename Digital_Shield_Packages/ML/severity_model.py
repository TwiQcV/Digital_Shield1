import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def train_severity_model(
    X_train_aug,
    X_test_encoded,
    y_train_aug,
    y_test,
    selected_columns: list = None,
    save_path: str = None
):
    """
    Train a Random Forest classification model and optionally save it.

    Parameters:
    -----------
    X_train_aug : pd.DataFrame
        Augmented training features
    X_test_encoded : pd.DataFrame
        Test features
    y_train_aug : pd.Series
        Augmented training labels
    y_test : pd.Series
        Test labels
    selected_columns : list, optional
        Column names to use (default: all)
    save_path : str, optional
        Path to save the trained model (e.g., 'models/severity_model.pkl')

    Returns:
    --------
    dict with keys:
        'model': trained model
        'accuracy': test accuracy
        'cv_mean': cross-validation mean accuracy
        'model_path': path where model was saved (if save_path provided)
    """

    # Select columns if specified
    if selected_columns is not None:
        if not isinstance(X_train_aug, np.ndarray):
            X_train_aug = X_train_aug[selected_columns].copy()
            X_test_encoded = X_test_encoded[selected_columns].copy()

    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print(f"\n🤖 Training model...")
    rf_model.fit(X_train_aug, y_train_aug)

    # Evaluate
    y_pred = rf_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        rf_model, X_train_aug, y_train_aug,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    results = {
        'model': rf_model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean()
    }

    # Save model if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_model, save_path)
        results['model_path'] = save_path
        print(f"✅ Model saved to: {save_path}")

    return results


def load_severity_model(model_path: str):
    """
    Load a previously trained model from disk.

    Parameters:
    -----------
    model_path : str
        Path to the saved model (e.g., 'models/severity_model.pkl')

    Returns:
    --------
    RandomForestClassifier: loaded model
    """

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


def predict_with_model(model_path: str, X_data, return_probabilities: bool = False):
    """
    Load model and make predictions.

    Parameters:
    -----------
    model_path : str
        Path to saved model
    X_data : pd.DataFrame or np.ndarray
        Data to predict on
    return_probabilities : bool
        If True, return prediction probabilities as well

    Returns:
    --------
    predictions : np.ndarray
        Predicted labels
    probabilities : np.ndarray (optional)
        Prediction probabilities if return_probabilities=True
    """

    model = load_severity_model(model_path)
    predictions = model.predict(X_data)

    if return_probabilities:
        probabilities = model.predict_proba(X_data)
        return predictions, probabilities

    return predictions


def evaluate_model(results):
    """
    Simple evaluation - returns accuracy only.

    Parameters:
    -----------
    results : dict
        Output from train_severity_model()

    Returns:
    --------
    float: accuracy score
    """
    return results['accuracy']
