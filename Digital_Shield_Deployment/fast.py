from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

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
    MODELS_DIR / "trained" / "financial_loss_xgboost.pkl",
    MODELS_DIR / "financial_loss_xgboost.pkl",
    MODELS_DIR / "financial_loss_model.pkl",
    ROOT / "Digital_Shield_Notebooks" / "models" / "financial_loss_xgboost.pkl",
    ROOT / "Digital_Shield_Notebooks" / "models" / "financial_loss_model.pkl",
]

# Globals

rag_system: Optional[Any] = None
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


# Load RAG (multiple strategies)

def load_rag():
    # 1) RAGSystem
    try:
        from Digital_Shield_Packages.RAG.main import RAGSystem  # type: ignore
        rag = RAGSystem()
        if hasattr(rag, "initialize"):
            rag.initialize(silent=True)
        return rag
    except Exception as e1:
        # 2) RAGPipeline / RagPipeline
        try:
            try:
                from Digital_Shield_Packages.RAG.pipeline import RAGPipeline  # type: ignore
                RagPipe = RAGPipeline
            except Exception:
                from Digital_Shield_Packages.RAG.pipeline import RagPipeline  # type: ignore
                RagPipe = RagPipeline
            rag = RagPipe()
            if hasattr(rag, "initialize"):
                rag.initialize(silent=True)
            return rag
        except Exception as e2:
            # 3) Retriever + Generator
            try:
                from Digital_Shield_Packages.RAG.config import RagConfig  # type: ignore
                from Digital_Shield_Packages.RAG.retriever import Retriever  # type: ignore
                from Digital_Shield_Packages.RAG.generator import Generator  # type: ignore

                try:
                    cfg = RagConfig()
                except Exception:
                    cfg = None

                retr = Retriever(cfg) if cfg is not None else Retriever()
                gen = Generator(cfg) if cfg is not None else Generator()

                class SimpleRAG:
                    def __init__(self, retriever, generator):
                        self.retriever = retriever
                        self.generator = generator

                    def ask(self, query: str) -> str:
                        if hasattr(self.retriever, "search"):
                            ctx = self.retriever.search(query, top_k=3)
                        elif hasattr(self.retriever, "retrieve"):
                            ctx = self.retriever.retrieve(query, top_k=3)
                        else:
                            ctx = []
                        if hasattr(self.generator, "generate"):
                            return self.generator.generate(query, ctx)
                        elif hasattr(self.generator, "generate_answer"):
                            return self.generator.generate_answer(query, ctx)
                        return "RAG generator has no generate() method."

                return SimpleRAG(retr, gen)
            except Exception as e3:
                print("[RAG] load failed:", e1, e2, e3)
                return None


# Schemas

class RAGRequest(BaseModel):
    query: str

class FinancialRequest(BaseModel):
    features: Dict[str, Any]


# Startup

@app.on_event("startup")
def on_startup():
    global rag_system, financial_artifact

    print("🔧 Initializing systems...")

    # -- RAG
    rag_system = load_rag()
    print("✅ RAG loaded" if rag_system is not None else "⚠️ RAG not loaded")

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


# Prediction helpers

def predict_financial_loss_row(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses saved artifact: sklearn preprocessor + XGBoost Booster trained on log1p(y).
    """
    if financial_artifact is None:
        raise HTTPException(status_code=503, detail="Financial model not loaded")

    model = financial_artifact["model"]            # xgboost.Booster
    preprocessor = financial_artifact["preprocessor"]
    feature_names = financial_artifact.get("feature_names", None)

    df = pd.DataFrame([features])

    # Transform with saved preprocessor (same as training)
    try:
        X_proc = preprocessor.transform(df)
    except Exception:
        # try aligning to training columns if feature_names_in_ available
        try:
            cols = getattr(preprocessor, "feature_names_in_", None)
            if cols is not None:
                X_proc = preprocessor.transform(df.reindex(columns=cols, fill_value=np.nan))
            else:
                raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # XGBoost inference (model trained on log1p -> invert)
    from xgboost import DMatrix
    dmat = DMatrix(X_proc, feature_names=feature_names)
    pred_log = model.predict(dmat)
    pred = float(np.expm1(pred_log[0]))
    pred = max(0.0, pred)
    return {"prediction": _to_py(pred)}

# Endpoints

@app.post("/rag_chat", tags=["rag"])
def rag_chat(req: RAGRequest):
    if rag_system is None:
        return {
            "answer": {
                "response": "RAG System not initialized. Please initialize first.",
                "error": True,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        }
    try:
        if hasattr(rag_system, "ask"):
            answer = rag_system.ask(req.query)
        elif hasattr(rag_system, "chat"):
            answer = rag_system.chat(req.query)
        else:
            answer = "RAG system loaded but no ask/chat method found."
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

@app.post("/predict_financial_loss", tags=["models"])
def predict_financial_loss_endpoint(req: FinancialRequest):
    return predict_financial_loss_row(req.features)
