# ===============================================
# severity_model.py
# Model: RandomForestClassifier
# Uses augmentation + splitting methods from external modules
# Model Accuracy: 0.5361
# ===============================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#  Import external methods
from augmentation import augment_data
from data_splitter import split_data


def train_severity_model():
    """
    Train a Random Forest model using preprocessed and augmented data.
    Returns:
        model: trained model
        X_test, y_test: for later evaluation
    """

    # 1️⃣ Get augmented data from augmentation module
    df = augment_data()  # balanced_df returned from augmentation.py

    # 2️⃣ Keep only selected features + target
    selected_columns = [
        'year',
        'number of affected users',
        'incident resolution time (in hours)',
        'data breach in gb',
        'severity_kmeans'
    ]
    df = df[selected_columns]

    # 3️⃣ Split dataset using split_data() method
    X_train, X_test, y_train, y_test = split_data(df, 'severity_kmeans')

    # 4️⃣ Train Random Forest model
    rf_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=200,
        max_depth=None,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 5️⃣ Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}\n")

    return rf_model


# -----------------------------------------------
# 🔮 Prediction Method
# -----------------------------------------------
def predict_severity(model, input_data):
    severity_map = {0:"Low", 1:"Medium", 2:"High",3:"Critical"}
    X_new = pd.DataFrame([input_data])
    pred_class = model.predict(X_new)[0]
    predicted_label = severity_map.get(pred_class, "Unknown")
    print(f"🔮 Predicted Severity: {predicted_label}")
    return predicted_label


# # -----------------------------------------------
# # Example Usage
# # -----------------------------------------------
# if __name__ == "__main__":
#     model, X_test, y_test = train_severity_model()

#     new_case = {
#         'year': 2025,
#         'number of affected users': 3000,
#         'incident resolution time (in hours)': 10,
#         'data breach in gb': 5.4
#     }

#     predict_severity(model, new_case)
