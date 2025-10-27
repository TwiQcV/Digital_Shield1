"""
Financial Loss Prediction Helpers

This module contains all the prediction-related functions and utilities
for the financial loss prediction model.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from fastapi import HTTPException


def calculate_severity(prediction: float) -> str:
    """Calculate severity based on prediction value"""
    if prediction < 5:
        return "Low"
    elif prediction < 20:
        return "Medium"
    elif prediction < 100:
        return "High"
    else:
        return "Critical"


def get_dataset_optimized_defaults(attack_type: str, target_industry: str, affected_users: int, data_breach_gb: float) -> dict:
    """
    Optimized defaults based on actual dataset frequency patterns.
    """
    # Most common resolution times by attack type (from dataset analysis)
    attack_resolution = {
        "Phishing": 37,           # Most common: 37 hours
        "Ransomware": 25,         # Common: 25 hours
        "Malware": 37,            # Common: 37 hours
        "DDoS": 37,              # Common: 37 hours
        "SQL Injection": 37,      # Common: 37 hours
        "Man-in-the-middle": 37   # Common: 37 hours
    }
    
    # Most common vulnerability-defense combinations by industry (from dataset)
    industry_patterns = {
        "IT": {"country": "UK", "vulnerability": "Zero Day", "defense": "Antivirus"},
        "Education": {"country": "UK", "vulnerability": "Zero Day", "defense": "VPN"},
        "Healthcare": {"country": "UK", "vulnerability": "Unpatched Software", "defense": "Antivirus"},
        "Banking": {"country": "UK", "vulnerability": "Zero Day", "defense": "Antivirus"},
        "Retail": {"country": "UK", "vulnerability": "Zero Day", "defense": "Encryption"},
        "Government": {"country": "UK", "vulnerability": "Unpatched Software", "defense": "Antivirus"},
        "Telecommunications": {"country": "UK", "vulnerability": "Zero Day", "defense": "Antivirus"}
    }
    
    # Calculate severity based on dataset patterns
    impact_score = (affected_users / 100000) * (data_breach_gb / 100)
    if impact_score < 0.1:
        severity = "Low"
    elif impact_score < 0.5:
        severity = "Medium"
    elif impact_score < 2.0:
        severity = "High"
    else:
        severity = "Critical"
    
    industry_defaults = industry_patterns.get(target_industry, industry_patterns["IT"])
    
    return {
        "resolution_time": attack_resolution.get(attack_type, 37),  # 37 is most common
        "country": "UK",  # Most frequent country
        "vulnerability_type": industry_defaults["vulnerability"],
        "defense_mechanism": industry_defaults["defense"],
        "severity": severity,
        "used_defaults": ["resolution_time", "country", "vulnerability_type", "defense_mechanism", "severity"]
    }


def apply_feature_engineering(df):
    """Apply the same feature engineering as in training"""
    df = df.copy()
    
    # Log transformations
    df["log_users"] = np.log1p(df["number of affected users"].fillna(0))
    df["log_breach"] = np.log1p(df["data breach in gb"].fillna(0))
    df["log_resolution_time"] = np.log1p(df["incident resolution time (in hours)"].fillna(0))
    
    # Interaction features
    df["impact_index"] = df["number of affected users"] * np.log1p(df["data breach in gb"].fillna(0))
    df["users_per_hour"] = df["number of affected users"] / (1.0 + df["incident resolution time (in hours)"].fillna(0))
    
    # Time features
    df["years_since_2010"] = pd.to_numeric(df["year"], errors="coerce") - 2010
    
    # Complex features
    df["severity_ratio"] = df["impact_index"] / (1.0 + df["users_per_hour"])
    df["complexity"] = df["log_users"] * df["log_breach"]
    
    return df


def apply_one_hot_encoding(df):
    """Apply one-hot encoding to match training format"""
    categorical_columns = ['country', 'attack type', 'target industry', 'security vulnerability type', 'defense mechanism used']
    
    for col in categorical_columns:
        if col in df.columns:
            for category in df[col].unique():
                if pd.notna(category):
                    dummy_col = f"{col}_{category.lower().replace(' ', '_').replace('-', '_')}"
                    df[dummy_col] = (df[col] == category).astype(int)
    
    df = df.drop(columns=categorical_columns)
    
    # Add all possible dummy columns with 0 values
    all_dummy_columns = [
        'country_australia', 'country_brazil', 'country_china', 'country_france',
        'country_germany', 'country_india', 'country_japan', 'country_russia',
        'country_uk', 'country_usa',
        'attack type_ddos', 'attack type_malware', 'attack type_man-in-the-middle',
        'attack type_phishing', 'attack type_ransomware', 'attack type_sql injection',
        'target industry_banking', 'target industry_education', 'target industry_government',
        'target industry_healthcare', 'target industry_it', 'target industry_retail',
        'target industry_telecommunications',
        'security vulnerability type_social engineering', 'security vulnerability type_unpatched software',
        'security vulnerability type_weak passwords', 'security vulnerability type_zero day',
        'defense mechanism used_ai based detection', 'defense mechanism used_antivirus',
        'defense mechanism used_encryption', 'defense mechanism used_firewall',
        'defense mechanism used_vpn'
    ]
    
    for col in all_dummy_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df.reindex(columns=sorted(df.columns), fill_value=0)


def _to_py(x):
    """Convert numpy types to Python native types"""
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass
    return x


def predict_financial_loss_minimal(features: Dict[str, Any], financial_artifact: dict) -> Dict[str, Any]:
    """
    Minimal 4-feature prediction with intelligent defaults to minimize accuracy loss.
    """
    if financial_artifact is None:
        raise HTTPException(status_code=503, detail="Financial model not loaded")

    model = financial_artifact["model"]
    preprocessor = financial_artifact["preprocessor"]
    feature_names = financial_artifact.get("feature_names", None)

    # Extract the 4 core features
    affected_users = features.get("number_of_affected_users", features.get("number of affected users", 100000))
    data_breach_gb = features.get("data_breach_size_gb", features.get("data breach in gb", 50.0))
    attack_type = features.get("attack_type", features.get("attack type", "Phishing"))
    target_industry = features.get("target_industry", features.get("target industry", "Retail"))
    
    # Calculate the most important engineered feature directly
    impact_index = affected_users * np.log1p(data_breach_gb)
    
    # Smart defaults based on attack type and industry patterns
    defaults = get_dataset_optimized_defaults(attack_type, target_industry, affected_users, data_breach_gb)
    
    # Create input data with optimized defaults
    input_data = {
        'year': 2024,
        'number of affected users': affected_users,
        'incident resolution time (in hours)': defaults['resolution_time'],
        'data breach in gb': data_breach_gb,
        'country': defaults['country'],
        'attack type': attack_type,
        'target industry': target_industry,
        'security vulnerability type': defaults['vulnerability_type'],
        'defense mechanism used': defaults['defense_mechanism'],
        'severity_kmeans': defaults['severity']
    }
    
    # Apply feature engineering (same as training)
    df = pd.DataFrame([input_data])
    df = apply_feature_engineering(df)
    
    # One-hot encode categorical variables
    df = apply_one_hot_encoding(df)
    
    # Transform and predict
    try:
        X_proc = preprocessor.transform(df)
        from xgboost import DMatrix
        dmat = DMatrix(X_proc, feature_names=feature_names)
        pred_log = model.predict(dmat)
        pred = float(np.expm1(pred_log[0]))
        pred = max(0.0, pred)
        
        return {
            "prediction": _to_py(pred),
            "severity": calculate_severity(pred),
            "confidence": "medium",  # Could add confidence scoring
            "used_defaults": defaults['used_defaults'],
            "impact_index": _to_py(impact_index)  # Show the calculated impact
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
