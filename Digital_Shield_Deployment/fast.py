from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Import RAG System
try:
    from Digital_Shield_Packages.RAG.main import RAGSystem
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG system not available: {e}")
    RAG_AVAILABLE = False


PURPLE_DARK = "#4B2E83"
PURPLE_LIGHT = "#B794F4"

app = FastAPI(
    title="Digital Shield API",
    version="0.3.0",
    swagger_ui_parameters={
        "docExpansion": "none",
        "defaultModelsExpandDepth": -1,
        "syntaxHighlight.theme": "tomorrow-night",
        "customCss": f"""
            .swagger-ui .topbar {{ background: {PURPLE_DARK} !important; }}
            .swagger-ui .topbar .link span {{ color: #fff !important; }}
            .swagger-ui .btn.execute.opblock-control__btn {{
                background: {PURPLE_DARK} !important; border-color: {PURPLE_DARK} !important;
            }}
            .swagger-ui .opblock.opblock-post {{
                border-color: {PURPLE_DARK} !important;
                background: rgba(183,148,244,0.08) !important;
            }}
            .swagger-ui .info .title {{ color: {PURPLE_DARK} !important; }}
            .swagger-ui .model-title, .swagger-ui .tab li.active a {{
                color: {PURPLE_DARK} !important;
            }}
            .swagger-ui .btn, .swagger-ui .opblock-summary-method {{
                background: {PURPLE_LIGHT} !important; border-color: {PURPLE_LIGHT} !important;
            }}
        """,
    },
)


# Paths

ROOT = Path(__file__).resolve().parents[1]      # Digital_Shield1/
MODELS_DIR = ROOT / "models"

FINLOSS_CANDIDATES: List[Path] = [
    ROOT / "Digital_Shield_Packages" / "models" / "financial_loss_xgboost.pkl",
    MODELS_DIR / "trained" / "financial_loss_xgboost.pkl",
    MODELS_DIR / "financial_loss_xgboost.pkl",
    MODELS_DIR / "financial_loss_model.pkl",
    ROOT / "Digital_Shield_Notebooks" / "models" / "financial_loss_xgboost.pkl",
    ROOT / "Digital_Shield_Notebooks" / "models" / "financial_loss_model.pkl",
]

# Globals

rag_system: Optional[RAGSystem] = None
financial_artifact: Optional[dict] = None   # {"model","preprocessor","feature_names",...}


# Helpers

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _to_py(x):
    try:
        import numpy as _np
        if isinstance(x, _np.generic):
            return x.item()
    except Exception:
        pass
    return x


# RAG System - Using existing RAG classes


# Schemas

class RAGRequest(BaseModel):
    query: str

class MinimalFinancialRequest(BaseModel):
    number_of_affected_users: int = 100000
    data_breach_size_gb: float = 50.0
    attack_type: str = "Phishing"
    target_industry: str = "Retail"


# Startup

@app.on_event("startup")
def on_startup():
    global rag_system, financial_artifact

    print("🔧 Initializing systems...")

    # -- RAG System initialization
    if RAG_AVAILABLE:
        try:
            rag_system = RAGSystem()
            if rag_system.initialize(silent=True):
                print("✅ RAG system initialized successfully")
            else:
                print("⚠️ RAG system initialization failed")
                rag_system = None
        except Exception as e:
            print(f"❌ RAG system error: {e}")
            rag_system = None
    else:
        print("⚠️ RAG system not available")

    # -- Financial Loss artifact
    fin_path = first_existing(FINLOSS_CANDIDATES)
    if not fin_path:
        print("❌ Financial model file not found in expected locations.")
        financial_artifact = None
    else:
        try:
            financial_artifact = joblib.load(fin_path)
            required = {"model", "preprocessor", "feature_names"}
            if not isinstance(financial_artifact, dict) or not required.issubset(financial_artifact.keys()):
                raise ValueError(f"Invalid artifact at {fin_path}. Missing keys {required}.")
            print(f"✅ Financial model loaded from: {fin_path}")
        except Exception as e:
            print(f"❌ Failed to load financial model: {e}")
            financial_artifact = None

    print("🚀 Startup complete.")


# Basic routes

@app.get("/", include_in_schema=False)
def root():
    return {"message": "welcome to our project Digital Shield"}

@app.get("/health", tags=["system"])
def health():
    return {
        "rag": rag_system is not None,
        "financial_model": financial_artifact is not None,
    }


# Import prediction helpers
from Digital_Shield_Packages.ML.prediction_helpers import predict_financial_loss_minimal


# Endpoints

@app.post("/rag_chat", tags=["rag"])
def rag_chat(req: RAGRequest):
    """
    RAG chat endpoint using the existing RAG system.
    Returns only response and suggested_queries fields.
    """
    if rag_system is None:
        return {
            "response": "RAG System not initialized. Please initialize first.",
            "suggested_queries": []
        }
    
    try:
        # Use the existing RAG system's query method
        result = rag_system.query(
            req.query,
            top_k=20,
            similarity_threshold=0.7
        )
        
        # Return only response and suggested_queries
        return {
            "response": result.get("response", ""),
            "suggested_queries": result.get("suggested_queries", [])
        }
        
    except Exception as e:
        return {
            "response": f"Error processing request: {e}",
            "suggested_queries": []
        }

@app.post("/predict_financial_loss", tags=["models"])
def predict_financial_loss_endpoint(req: MinimalFinancialRequest):
    """
    Financial loss prediction endpoint - simplified to use 4 core features directly.
    Uses intelligent defaults based on dataset patterns to maintain accuracy.
    """
    features = {
        "number_of_affected_users": req.number_of_affected_users,
        "data_breach_size_gb": req.data_breach_size_gb,
        "attack_type": req.attack_type,
        "target_industry": req.target_industry
    }
    return predict_financial_loss_minimal(features, financial_artifact)

@app.post("/predict_financial_loss_simple", tags=["models"])
def predict_financial_loss_simple_endpoint(req: MinimalFinancialRequest):
    """
    Simplified financial loss prediction endpoint - only requires 4 core features.
    Uses intelligent defaults based on dataset patterns to maintain accuracy.
    """
    features = {
        "number_of_affected_users": req.number_of_affected_users,
        "data_breach_size_gb": req.data_breach_size_gb,
        "attack_type": req.attack_type,
        "target_industry": req.target_industry
    }
    return predict_financial_loss_minimal(features, financial_artifact)
