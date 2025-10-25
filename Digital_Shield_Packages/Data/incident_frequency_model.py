# file: incident_frequency_model.py
# ===========================================
#  Incident Frequency Prediction (Aggregated & Safe)
# RandomForestRegressor with One-Hot Encoding
# Smart version: saves and loads trained model
# ===========================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import your existing split method
from data_splitter import split_data  # <-- make sure the path is correct

MODEL_FILE = "incident_model.pkl"  # File name where the trained model will be saved

def run_incident_pipeline(df, force_train=False):
    """
    Full pipeline with  model saving/loading.
    This function:
    - Aggregates and encodes data
    - Splits data using existing method
    - Trains RandomForestRegressor (once)
    - Uses saved model on subsequent runs to avoid retraining
    - Returns evaluation metrics and predictions

    Args:
        df (DataFrame): raw input dataset
        force_train (bool): if True, retrain the model even if saved model exists

    Returns:
        model: trained RandomForestRegressor
        X_test, y_test: test dataset
        y_pred: predictions on X_test
        mae, r2: evaluation metrics
    """
    df_model = df.copy()  # Copy dataframe to avoid changing original data

    # 1️⃣ Add incident_count column
    # Each row represents 1 incident
    df_model['incident_count'] = 1

    # 2️⃣ Define numeric and categorical features
    numeric_features = [
        'number of affected users',
        'incident resolution time (in hours)',
        'data breach in gb'
    ]
    country_cols = [col for col in df_model.columns if col.startswith('country_')]
    categorical_features = ['severity_kmeans']

    # 3️⃣ Aggregate data
    # Group by year, country columns, and severity
    # Sum incident_count and take mean for numeric features
    group_cols = ['year'] + country_cols + categorical_features
    agg_dict = {'incident_count': 'sum'}
    for col in numeric_features:
        agg_dict[col] = 'mean'

    incident_df = df_model.groupby(group_cols).agg(agg_dict).reset_index()

    # 4️⃣ One-Hot Encode categorical columns
    # Convert categorical severity_kmeans to numeric columns
    incident_df_encoded = pd.get_dummies(incident_df, columns=categorical_features, drop_first=True)

    # 5️⃣ Split data using your existing method
    # split_data should return: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(incident_df_encoded, target_column='incident_count')

    # 6️⃣ Check if trained model already exists
    if os.path.exists(MODEL_FILE) and not force_train:
        # Load the saved model from disk
        print("⚡ Loading saved model from disk...")
        model = joblib.load(MODEL_FILE)
    else:
        # Train new RandomForestRegressor
        print("⚡ Training new model...")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        # Save the trained model to disk for future use
        joblib.dump(model, MODEL_FILE)
        print(f"✅ Model saved to {MODEL_FILE}")

    # 7️⃣ Predict on test set using trained or loaded model
    y_pred = model.predict(X_test)

    # 8️⃣ Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)             # R² score
    print(" Incident Frequency Model Results (RandomForest, Aggregated & Safe)")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Optional: display sample predictions
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred, 2)}).head(10)
    print("\nSample Predictions:")
    print(comparison_df)

    return model, X_test, y_test, y_pred, mae, r2
