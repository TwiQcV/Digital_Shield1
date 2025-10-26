import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from xgboost import DMatrix, train as xgb_train


class ModelSaver:
    """Save and load trained XGBoost models with metadata"""

    @staticmethod
    def save_model(bst, preprocessor, feature_names, metrics, model_path: str, metadata: Dict = None) -> Dict:
        """
        Save XGBoost model, preprocessor, and metadata
        Args:
            bst: Trained XGBoost model
            preprocessor: Fitted ColumnTransformer preprocessor
            feature_names: List of feature names
            metrics: Dictionary of evaluation metrics
            model_path: Path to save model
            metadata: Additional metadata to store
        Returns:
            Dictionary with save information
        """
        # Create model directory
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Prepare model artifact
        model_artifact = {
            "model": bst,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # Save using joblib (better for sklearn objects)
        try:
            joblib.dump(model_artifact, model_path)
            print(f"✅ Model saved successfully to: {model_path}")

            # Also save model summary
            summary_path = model_path.replace(".pkl", "_summary.txt")
            ModelSaver._save_summary(model_path, metrics, summary_path)

            return {
                "success": True,
                "model_path": model_path,
                "summary_path": summary_path,
                "file_size_mb": Path(model_path).stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def load_model(model_path: str) -> Dict:
        """
        Load saved XGBoost model with all components
        Args:
            model_path: Path to saved model
        Returns:
            Dictionary containing model, preprocessor, feature_names, and metrics
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        try:
            model_artifact = joblib.load(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
            return model_artifact
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise

    @staticmethod
    def _save_summary(model_path: str, metrics: Dict, summary_path: str) -> None:
        """Save model summary as text file"""
        summary_text = f"""
    XGBoost Financial Loss Prediction Model Summary
    {'='*60}
    Model Path: {model_path}
    Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Performance Metrics:
    {'-'*60}
    MAE:        {metrics.get('MAE', 'N/A'):,.3f}
    Median AE:  {metrics.get('Median_AE', 'N/A'):,.3f}
    RMSE:       {metrics.get('RMSE', 'N/A'):,.3f}
    R²:         {metrics.get('R2', 'N/A'):,.4f}
    sMAPE:      {metrics.get('sMAPE', 'N/A'):,.2f}%
    Note: Use ModelSaver.load_model() to restore this model
    """
        Path(summary_path).write_text(summary_text)


class DataLoader:
    """Load and validate financial loss dataset"""

    @staticmethod
    def load_data(csv_filenames: List[str] = None) -> pd.DataFrame:
        """
        Load CSV file from specified filenames or defaults
        Args:
        csv_filenames: List of filenames to try. Defaults to common variations.
        Returns:
        DataFrame with raw data
        Raises:
        AssertionError: If no CSV file found
        """
    # Get project root (two levels up from current file)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if csv_filenames is None:
        # Build paths using os.path.join for cross-platform compatibility
        # Try multiple possible directory structures
            possible_data_dirs = [
            os.path.join(project_root, 'Digital_Shield_data', 'proccesed'),
            os.path.join(project_root, 'Digital_Shield1', 'Digital_Shield_data', 'proccesed'),
            os.path.join(project_root, '..', 'Digital_Shield_data', 'proccesed'),  # One level up
            os.path.join(project_root, '..', 'Digital_Shield1', 'Digital_Shield_data', 'proccesed'),  # One level up
        ]

        csv_filenames = []
        for data_dir in possible_data_dirs:
            csv_filenames.extend([
                os.path.join(data_dir, 'Data Augmentetion.csv'),
                os.path.join(data_dir, 'Data Augmentation.csv'),
            ])

        # Add relative paths and current directory
        csv_filenames.extend([
            "./Data Augmentetion.csv",
            "./Data Augmentation.csv",
            "Data Augmentetion.csv",
            "Data Augmentation.csv"
        ])

    # Debug: Print what paths we're checking

        csv = next((Path(p) for p in csv_filenames if Path(p).exists()), None)
        assert csv is not None, f"CSV file not found. Tried: {csv_filenames}"

        print(f"📂 Loading data from: {csv}")
        return pd.read_csv(csv)

    @staticmethod
    def extract_target(df: pd.DataFrame, target_col_names: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract target column and handle missing values
        Args:
            df: Input dataframe
            target_col_names: Possible target column names. Defaults to common variations.
        Returns:
            (X, y) tuple with features and target
        """
        if target_col_names is None:
            target_col_names = ["Financial Loss (in Million $)", "financial loss (in million $)"]

        target = next((t for t in target_col_names if t in df.columns), None)
        assert target is not None, f"Target not found. Tried: {target_col_names}"

        X = df.drop(columns=[target]).copy()
        y = pd.to_numeric(df[target], errors="coerce").copy()

        # Drop NaN target rows
        mask_ok = y.notna()
        X, y = X.loc[mask_ok].reset_index(drop=True), y.loc[mask_ok].reset_index(drop=True)

        return X, y


class DataPreprocessor:
    """Data cleaning and validation"""

    @staticmethod
    def clean_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Clean numeric columns and validate target
        Args:
            X: Feature dataframe
            y: Target series
        Returns:
            (X_clean, y_clean) tuple
        """
        X = X.copy()
        y = y.copy()

        # Convert year to numeric
        if "year" in X.columns:
            X["year"] = pd.to_numeric(X["year"], errors="coerce")

        # Ensure non-negative values for specific columns
        for col in ["number of affected users", "data breach in gb", "incident resolution time (in hours)"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(lower=0)

        # Validate target
        if (y < 0).any():
            raise ValueError("Negative values in target — check data before using log1p")

        return X, y


class FeatureEngineer:
    """Feature engineering for financial loss prediction"""

    @staticmethod
    def add_if(df: pd.DataFrame, cols: List[str], name: str, func) -> None:
        """
        Conditionally add engineered feature to dataframe
        Args:
            df: Input dataframe (modified in-place)
            cols: Required columns for feature
            name: New feature name
            func: Function to compute feature from columns
        """
        if all(c in df.columns for c in cols) and name not in df.columns:
            try:
                df[name] = func(*(df[c].fillna(0) for c in cols))
                df[name] = pd.to_numeric(df[name], errors="coerce")
            except Exception:
                df[name] = np.nan

    @staticmethod
    def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        Args:
            X: Input dataframe
        Returns:
            Dataframe with engineered features added
        """
        X = X.copy()

        # Log transformations
        FeatureEngineer.add_if(X, ["number of affected users"], "log_users",
                              lambda u: np.log1p(u))
        FeatureEngineer.add_if(X, ["data breach in gb"], "log_breach",
                              lambda g: np.log1p(g))
        FeatureEngineer.add_if(X, ["incident resolution time (in hours)"], "log_resolution_time",
                              lambda t: np.log1p(t))

        # Interaction features
        FeatureEngineer.add_if(X, ["number of affected users", "data breach in gb"], "impact_index",
                              lambda u, g: u * np.log1p(g))
        FeatureEngineer.add_if(X, ["number of affected users", "incident resolution time (in hours)"],
                              "users_per_hour", lambda u, t: u / (1.0 + t))

        # Time features
        FeatureEngineer.add_if(X, ["year"], "years_since_2010",
                              lambda yr: pd.to_numeric(yr, errors="coerce") - 2010)

        # Complex features
        if {"impact_index", "users_per_hour"} <= set(X.columns):
            X["severity_ratio"] = X["impact_index"] / (1.0 + X["users_per_hour"])

        if {"log_users", "log_breach"} <= set(X.columns):
            X["complexity"] = X["log_users"] * X["log_breach"]

        # Remove infinities
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        return X


class PreprocessingPipeline:
    """Sklearn preprocessing pipeline for numeric and categorical features"""

    @staticmethod
    def build_pipeline(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
        """
        Build preprocessing pipeline (fit on training data)
        Args:
            X_train: Training features
        Returns:
            (fitted_preprocessor, feature_names) tuple
        """
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_transformer = Pipeline([("imp", SimpleImputer(strategy="median"))])

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        cat_transformer = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", ohe)
        ])

        preprocess = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_cols),
                ("cat", cat_transformer, cat_cols)
            ],
            remainder="drop",
            sparse_threshold=0.0
        )

        # Fit to get feature names
        preprocess.fit(X_train)

        try:
            ohe_step = preprocess.named_transformers_["cat"].named_steps["enc"]
            ohe_names = ohe_step.get_feature_names_out(cat_cols).tolist()
        except Exception:
            ohe_names = []

        feature_names = num_cols + ohe_names

        return preprocess, feature_names


class XGBoostTrainer:
    """XGBoost model training with cross-validation and no data leakage"""

    # Default hyperparameters
    DEFAULT_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.02,
        "max_depth": 6,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.2,
        "alpha": 0.15,
        "gamma": 0.2,
        "tree_method": "hist",
        "seed": 42
    }

    def __init__(self, params: Dict = None, n_folds: int = 5, early_stopping_rounds: int = 100,
                 num_boost_round: int = 3000, verbose: bool = False):
        """
        Initialize XGBoost trainer
        Args:
            params: XGBoost parameters (uses defaults if None)
            n_folds: Number of cross-validation folds
            early_stopping_rounds: Early stopping patience
            num_boost_round: Maximum boosting rounds
            verbose: Print training progress
        """
        self.params = params or self.DEFAULT_PARAMS
        self.n_folds = n_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose = verbose
        self.best_iters = []

    def train_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Train XGBoost with k-fold cross-validation (no data leakage)
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
        Returns:
            (test_predictions, metadata) tuple
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        test_preds = np.zeros(len(X_test))
        self.best_iters = []
        fold_details = []
        best_model = None
        best_preprocessor = None
        best_feature_names = None

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
            X_tr, X_va = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
            y_tr, y_va = y_train.iloc[tr_idx].copy(), y_train.iloc[va_idx].copy()

            # Build preprocessing inside fold (prevents leakage)
            preprocess, feature_names = PreprocessingPipeline.build_pipeline(X_tr)

            # Transform all sets
            X_tr_p = preprocess.transform(X_tr)
            X_va_p = preprocess.transform(X_va)
            X_te_p = preprocess.transform(X_test)

            # Log-transform target
            y_tr_log = np.log1p(y_tr.values)
            y_va_log = np.log1p(y_va.values)

            # Create DMatrix objects
            dtr = DMatrix(X_tr_p, label=y_tr_log, feature_names=feature_names)
            dva = DMatrix(X_va_p, label=y_va_log, feature_names=feature_names)
            dte = DMatrix(X_te_p, feature_names=feature_names)

            # Train XGBoost
            bst = xgb_train(
                params=self.params,
                dtrain=dtr,
                num_boost_round=self.num_boost_round,
                evals=[(dtr, "train"), (dva, "valid")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose
            )

            best_it = int(bst.best_iteration) if bst.best_iteration else self.num_boost_round - 1
            self.best_iters.append(best_it + 1)

            # Predict and inverse log-transform
            pred_log = bst.predict(dte, iteration_range=(0, best_it + 1))
            fold_preds = np.expm1(pred_log)
            fold_preds = np.clip(fold_preds, 0, float(y_train.max()) * 10)

            test_preds += fold_preds / self.n_folds
            fold_details.append({
                "fold": fold,
                "best_iteration": best_it + 1,
                "train_samples": len(tr_idx),
                "val_samples": len(va_idx)
            })

            # Save best model from fold 1 (for model persistence)
            if fold == 1:
                best_model = bst
                best_preprocessor = preprocess
                best_feature_names = feature_names

        metadata = {
            "best_iters": self.best_iters,
            "avg_best_iter": int(np.mean(self.best_iters)),
            "fold_details": fold_details,
            "best_model": best_model,
            "best_preprocessor": best_preprocessor,
            "best_feature_names": best_feature_names
        }

        return test_preds, metadata


class ModelEvaluator:
    """Evaluate regression model performance"""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute comprehensive regression metrics
        Args:
            y_true: True values
            y_pred: Predicted values
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Median_AE": median_ae,
            "sMAPE": smape
        }

    @staticmethod
    def get_worst_predictions(y_true: np.ndarray, y_pred: np.ndarray, n: int = 10) -> pd.DataFrame:
        """
        Get top N worst predictions by absolute error
        Args:
            y_true: True values
            y_pred: Predicted values
            n: Number of worst predictions to return
        Returns:
            DataFrame with worst predictions
        """
        err = np.abs(y_true - y_pred)
        bad_idx = np.argsort(-err)[:n]

        return pd.DataFrame({
            "y_true": y_true[bad_idx],
            "y_pred": y_pred[bad_idx],
            "abs_err": err[bad_idx]
        })

    @staticmethod
    def print_results(metrics: Dict, worst_preds: pd.DataFrame, best_iters: List[int]) -> None:
        """
        Print formatted evaluation results
        Args:
            metrics: Dictionary of metrics
            worst_preds: DataFrame of worst predictions
            best_iters: List of best iterations per fold
        """
        print("✅ XGBoost — no data leakage version")
        print("Best trees per fold:", best_iters, "| avg:", int(np.mean(best_iters)))
        print("-" * 74)
        print("📊 Test (Holdout) Metrics")
        print(f"{'MAE':<20}: {metrics['MAE']:,.3f}")
        print(f"{'Median AE':<20}: {metrics['Median_AE']:,.3f}")
        print(f"{'RMSE':<20}: {metrics['RMSE']:,.3f}")
        print(f"{'R²':<20}: {metrics['R2']:,.4f}")
        print(f"{'sMAPE':<20}: {metrics['sMAPE']:,.2f}%")
        print("\nTop 10 worst absolute errors on holdout:")
        print(worst_preds.to_string(index=False))


class FinancialLossPipeline:
    """
    Complete end-to-end pipeline for financial loss prediction
    Usage:
        pipeline = FinancialLossPipeline(model_save_path="models/financial_loss_model.pkl")
        results = pipeline.run()
    """

    def __init__(self, test_size: float = 0.2, xgb_params: Dict = None,
                 n_folds: int = 5, verbose: bool = False,
                 model_save_path: str = "models/financial_loss_model.pkl"):
        """
        Initialize pipeline
        Args:
            test_size: Test set fraction
            xgb_params: XGBoost parameters
            n_folds: Number of cross-validation folds
            verbose: Print training progress
            model_save_path: Path to save trained model
        """
        self.test_size = test_size
        self.xgb_params = xgb_params
        self.n_folds = n_folds
        self.verbose = verbose
        self.model_save_path = model_save_path
        self.results = {}

    def run(self) -> Dict:
        """
        Execute complete pipeline
        Returns:
            Dictionary with results and metrics
        """
        # Load data
        print("📂 Loading data...")
        df = DataLoader.load_data()
        X, y = DataLoader.extract_target(df)

        # Clean data
        print("🧹 Cleaning data...")
        X, y = DataPreprocessor.clean_data(X, y)

        # Feature engineering
        print("🔧 Engineering features...")
        X = FeatureEngineer.engineer_features(X)

        # Split data
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=None
        )

        # Train and predict
        print(f"🚀 Training XGBoost with {self.n_folds}-fold CV...")
        trainer = XGBoostTrainer(
            params=self.xgb_params,
            n_folds=self.n_folds,
            verbose=self.verbose
        )
        test_preds, metadata = trainer.train_and_predict(X_train, y_train, X_test)

        # Evaluate
        print("📊 Evaluating model...")
        metrics = ModelEvaluator.compute_metrics(y_test.values, test_preds)
        worst_preds = ModelEvaluator.get_worst_predictions(y_test.values, test_preds)

        # Print results
        ModelEvaluator.print_results(metrics, worst_preds, metadata["best_iters"])

        # Save model
        print("\n💾 Saving model...")
        save_info = ModelSaver.save_model(
            bst=metadata["best_model"],
            preprocessor=metadata["best_preprocessor"],
            feature_names=metadata["best_feature_names"],
            metrics=metrics,
            model_path=self.model_save_path,
            metadata={
                "test_size": self.test_size,
                "n_folds": self.n_folds,
                "features_count": len(X_train.columns),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
        )

        # Store results
        self.results = {
            "metrics": metrics,
            "worst_predictions": worst_preds,
            "test_predictions": test_preds,
            "y_test": y_test,
            "metadata": metadata,
            "save_info": save_info
        }

        return self.results

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using saved model
        Args:
            X_new: New data for prediction
        Returns:
            Predicted values
        """
        # Load model
        model_artifact = ModelSaver.load_model(self.model_save_path)
        bst = model_artifact["model"]
        preprocessor = model_artifact["preprocessor"]
        feature_names = model_artifact["feature_names"]

        # Preprocess
        X_processed = preprocessor.transform(X_new)

        # Create DMatrix
        dx = DMatrix(X_processed, feature_names=feature_names)

        # Predict and inverse log-transform
        pred_log = bst.predict(dx)
        predictions = np.expm1(pred_log)

        return predictions
