import sys
import argparse
from pathlib import Path

# Import from Data package
from Data.data_config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    TEXT_COLUMNS,
    SPELLING_MAPS,
    DTYPE_MAP,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    NUMERIC_IMPUTATION_STRATEGY,
    CATEGORICAL_IMPUTATION_STRATEGY
)

from Data.data_cleaner import DataCleaner
from Data.data_splitter import split_data
from Data.encoding import apply_one_hot_encoding
from Data.augmentation import augment_data
from Data.clusterer import add_severity_to_dataset

# Import ML modules
from ML.severity_model import (
    train_severity_model,
    load_severity_model,
    predict_with_model,
    evaluate_model
)

from ML.config import (
    MODEL_SAVE_PATH,
    SELECTED_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)

from ML.fanancial_loss_model import (
    FinancialLossPipeline,
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelSaver,
    XGBoostTrainer,
    PreprocessingPipeline,
    ModelEvaluator
)

# Configuration
DATA_PATH = "/home/nawaf/code/TwiQcV/Digital_Shield1/Digital_Shield_data/proccesed/Data Augmentetion.csv"
MODEL_SAVE_PATH_FINANCIAL = "models/financial_loss_xgboost.pkl"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_FOLDS = 5
RANDOM_STATE = 42


def validate_paths(raw_path: str) -> bool:
    """Validate that data paths exist."""
    if not Path(raw_path).exists():
        print(f"❌ Error: Raw data not found at {raw_path}")
        return False
    print(f"✅ Data path verified: {raw_path}")
    return True


def model_exists(model_path: str) -> bool:
    """Check if model file exists."""
    return Path(model_path).exists()


def run_data_cleaning(raw_path: str, cleaned_path: str) -> object:
    """Execute data cleaning pipeline."""
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)

    try:
        cleaner = DataCleaner(raw_path)

        cleaned_df = cleaner.run_full_pipeline(
            text_cols=TEXT_COLUMNS,
            spelling_maps=SPELLING_MAPS,
            dtype_map=DTYPE_MAP,
            numeric_cols=NUMERIC_COLUMNS,
            categorical_cols=CATEGORICAL_COLUMNS,
            drop_cols=DROP_COLUMNS
        )

        validation = cleaner.validate_dataset()
        print(f"\n📊 Dataset Validation:")
        print(f"   Total Rows: {validation['total_rows']}")
        print(f"   Total Columns: {validation['total_cols']}")
        print(f"   Null Values: {validation['null_values']}")

        Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"\n💾 Cleaned data saved to: {cleaned_path}")

        return cleaned_df

    except Exception as e:
        print(f"❌ Data cleaning failed: {str(e)}")
        sys.exit(1)


def run_clustering(df, feature_column: str = "financial loss (in million $)") -> object:
    """Execute severity clustering."""
    print("\n" + "="*70)
    print("STEP 1.5: SEVERITY CLUSTERING")
    print("="*70)

    try:
        df, severity_stats = add_severity_to_dataset(
            df,
            feature_column=feature_column,
            n_clusters=4,
            random_state=RANDOM_STATE
        )
        return df

    except Exception as e:
        print(f"❌ Severity clustering failed: {str(e)}")
        sys.exit(1)


def run_data_splitting(df, target_col: str = "severity_kmeans", test_size: float = None):
    """Execute data splitting."""
    print("\n" + "="*70)
    print("STEP 2: DATA SPLITTING")
    print("="*70)

    if test_size is None:
        test_size = TEST_SIZE

    try:
        X_train, X_test, y_train, y_test = split_data(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=RANDOM_STATE
        )

        print(f"\n✅ Data split successfully:")
        print(f"   Training set: {X_train.shape} rows × {X_train.shape} features")
        print(f"   Test set: {X_test.shape} rows × {X_test.shape} features")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"❌ Data splitting failed: {str(e)}")
        sys.exit(1)


def run_encoding(X_train, X_test):
    """Execute feature encoding."""
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENCODING")
    print("="*70)

    try:
        X_train_encoded, X_test_encoded = apply_one_hot_encoding(X_train, X_test)

        print(f"\n✅ Encoding completed:")
        print(f"   Training set: {X_train_encoded.shape}")
        print(f"   Test set: {X_test_encoded.shape}")

        return X_train_encoded, X_test_encoded

    except Exception as e:
        print(f"❌ Feature encoding failed: {str(e)}")
        sys.exit(1)


def run_augmentation(X_train_encoded, y_train):
    """Execute data augmentation."""
    print("\n" + "="*70)
    print("STEP 4: DATA AUGMENTATION (SMOTE)")
    print("="*70)

    try:
        X_train_aug, y_train_aug, balanced_df = augment_data(X_train_encoded, y_train)

        print(f"\n✅ Data augmentation completed:")
        print(f"   Augmented training set: {X_train_aug.shape} rows × {X_train_aug.shape} features")

        return X_train_aug, y_train_aug, balanced_df

    except Exception as e:
        print(f"❌ Data augmentation failed: {str(e)}")
        sys.exit(1)


def run_model_training(X_train_aug, X_test_encoded, y_train_aug, y_test, save_path: str = None):
    """Execute severity model training and save."""
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING")
    print("="*70)

    try:
        results = train_severity_model(
            X_train_aug=X_train_aug,
            X_test_encoded=X_test_encoded,
            y_train_aug=y_train_aug,
            y_test=y_test,
            selected_columns=SELECTED_FEATURES,
            save_path=save_path
        )

        accuracy = evaluate_model(results)

        print(f"\n✅ Model trained successfully!")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   CV Mean: {results['cv_mean']:.4f}")

        if save_path:
            print(f"   💾 Model saved: {results.get('model_path', save_path)}")

        return results

    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        sys.exit(1)


def run_model_inference(model_path: str, X_test_encoded):
    """Load and use saved model for inference."""
    print("\n" + "="*70)
    print("STEP 5: MODEL INFERENCE (USING SAVED MODEL)")
    print("="*70)

    try:
        model = load_severity_model(model_path)
        predictions = predict_with_model(model_path, X_test_encoded[SELECTED_FEATURES])

        print(f"\n✅ Model loaded and predictions made!")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Unique classes: {len(set(predictions))}")

        return predictions

    except Exception as e:
        print(f"❌ Model inference failed: {str(e)}")
        sys.exit(1)


def run_financial_loss_training(test_size: float = DEFAULT_TEST_SIZE,
                                n_folds: int = DEFAULT_N_FOLDS,
                                verbose: bool = False) -> dict:
    """Execute financial loss prediction model training with model saving."""
    print("\n" + "="*70)
    print("STEP 6: FINANCIAL LOSS PREDICTION MODEL TRAINING")
    print("="*70)

    try:
        pipeline = FinancialLossPipeline(
            test_size=test_size,
            n_folds=n_folds,
            verbose=verbose,
            model_save_path=MODEL_SAVE_PATH_FINANCIAL
        )

        results = pipeline.run()

        print("\n" + "="*70)
        print("✅ FINANCIAL LOSS MODEL TRAINED SUCCESSFULLY")
        print("="*70)
        print(f"\n📊 Financial Loss Model Metrics:")
        print(f"  MAE (Mean Absolute Error):    {results['metrics']['MAE']:,.3f}M$")
        print(f"  RMSE (Root Mean Squared):     {results['metrics']['RMSE']:,.3f}M$")
        print(f"  Median Absolute Error:        {results['metrics']['Median_AE']:,.3f}M$")
        print(f"  R² Score:                     {results['metrics']['R2']:,.4f}")
        print(f"  sMAPE:                        {results['metrics']['sMAPE']:,.2f}%")

        print(f"\n💾 Financial Loss Model Information:")
        print(f"  Save Path:    {results['save_info']['model_path']}")
        print(f"  File Size:    {results['save_info']['file_size_mb']:.2f} MB")
        print(f"  Summary:      {results['save_info']['summary_path']}")

        print(f"\n📈 Training Configuration:")
        print(f"  Test Size:            {test_size*100:.1f}%")
        print(f"  CV Folds:             {n_folds}")
        print(f"  Best Iterations:      {results['metadata']['best_iters']}")
        print(f"  Average Best Iter:    {results['metadata']['avg_best_iter']}")

        return results

    except Exception as e:
        print(f"❌ Financial loss model training failed: {str(e)}")
        sys.exit(1)



def run_financial_loss_inference(test_size: float = DEFAULT_TEST_SIZE) -> dict:
    """Execute financial loss model inference using saved model."""
    print("\n" + "="*70)
    print("STEP 6: FINANCIAL LOSS MODEL INFERENCE (USING SAVED MODEL)")
    print("="*70)

    if not model_exists(MODEL_SAVE_PATH_FINANCIAL):
        print(f"❌ Financial loss model not found at: {MODEL_SAVE_PATH_FINANCIAL}")
        print(f"   Train model first: python main.py")
        sys.exit(1)

    try:
        # Load data
        print("📂 Loading data for financial loss inference...")
        df = DataLoader.load_data([DATA_PATH])
        X, y = DataLoader.extract_target(df)

        # Preprocess
        print("🧹 Cleaning data...")
        X, y = DataPreprocessor.clean_data(X, y)

        print("🔧 Engineering features...")
        X = FeatureEngineer.engineer_features(X)

        # Split to get test data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

        # Make predictions
        print("🔮 Making predictions with saved financial loss model...")
        pipeline = FinancialLossPipeline(model_save_path=MODEL_SAVE_PATH_FINANCIAL)
        predictions = pipeline.predict(X_test)

        # Evaluate
        metrics = ModelEvaluator.compute_metrics(y_test.values, predictions)
        worst_preds = ModelEvaluator.get_worst_predictions(y_test.values, predictions, n=5)

        print("\n" + "="*70)
        print("✅ FINANCIAL LOSS MODEL INFERENCE COMPLETED")
        print("="*70)
        print(f"\n📊 Financial Loss Inference Metrics:")
        print(f"  MAE:    {metrics['MAE']:,.3f}M$")
        print(f"  RMSE:   {metrics['RMSE']:,.3f}M$")
        print(f"  R²:     {metrics['R2']:,.4f}")
        print(f"  sMAPE:  {metrics['sMAPE']:,.2f}%")

        print(f"\n📈 Sample Predictions (first 5):")
        print(f"  Actual vs Predicted:")
        for i in range(min(5, len(y_test))):
            print(f"    {i+1}. Actual: ${y_test.values[i]:,.2f}M → Predicted: ${predictions[i]:,.2f}M")

        print(f"\n⚠️  Top 5 Worst Predictions:")
        print(worst_preds.to_string(index=False))

        return {
            "metrics": metrics,
            "predictions": predictions,
            "y_test": y_test,
            "worst_predictions": worst_preds
        }

    except Exception as e:
        print(f"❌ Financial loss model inference failed: {str(e)}")
        sys.exit(1)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Digital Shield Cybersecurity Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline + save model
  python main.py --inference        # Use saved model (no retraining)
  python main.py --skip-training    # Full pipeline but don't train
  python main.py --test-size 0.2    # Use 20% test split
        """
    )

    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip SMOTE augmentation step"
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help="Use saved model (skip retraining)"
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Test set proportion (default: 0.3)"
    )

    parser.add_argument(
    "--financial-only",
    action="store_true",
    help="Run financial loss model training only"
)

    parser.add_argument(
    "--n-folds",
    type=int,
    default=DEFAULT_N_FOLDS,
    help="Number of cross-validation folds (default: 5)"
    )

    parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose output during training"
    )

    parser.add_argument(
    "--skip-financial",
    action="store_true",
    help="Skip financial loss model training"
    )
    args = parser.parse_args()



    print("\n" + "="*70)
    print("🛡️  DIGITAL SHIELD CYBERSECURITY DATA PIPELINE")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"   Selected features: {len(SELECTED_FEATURES)} features")
    print(f"   Model save path: {MODEL_SAVE_PATH}")
    print(f"   Inference mode: {args.inference}")

    if args.financial_only:
        print("\n💡 Running financial loss model training only...")

        if model_exists(MODEL_SAVE_PATH_FINANCIAL) and args.inference:
            results_financial = run_financial_loss_inference(test_size=args.test_size or DEFAULT_TEST_SIZE)
        else:
            results_financial = run_financial_loss_training(
                test_size=args.test_size or DEFAULT_TEST_SIZE,
                n_folds=args.n_folds,
                verbose=args.verbose
            )
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY (FINANCIAL LOSS MODEL ONLY)")
        print("="*70)
        print(f"\n💡 Next steps:")
        print(f"   - Train full pipeline: python main.py")
        print(f"   - Use saved model: python main.py --inference")
        print("\n")
        return

    # MODE 1: Use saved model (no retraining)
    if args.inference:
        if not model_exists(MODEL_SAVE_PATH):
            print(f"\n❌ Model not found at: {MODEL_SAVE_PATH}")
            print(f"   Train model first: python main.py")
            sys.exit(1)

        if not validate_paths(RAW_DATA_PATH):
            sys.exit(1)

        cleaned_df = run_data_cleaning(RAW_DATA_PATH, CLEANED_DATA_PATH)
        cleaned_df = run_clustering(cleaned_df)
        X_train, X_test, y_train, y_test = run_data_splitting(cleaned_df)
        X_train_encoded, X_test_encoded = run_encoding(X_train, X_test)

        predictions = run_model_inference(MODEL_SAVE_PATH, X_test_encoded)

        if not args.skip_financial:
            results_financial = run_financial_loss_inference(test_size=args.test_size or DEFAULT_TEST_SIZE)

        print(f"\n✅ INFERENCE COMPLETED (BOTH MODELS)")
        print(f"\n📊 Results Summary:")
        print(f"   Severity model predictions: {len(predictions_severity)} samples")
        if not args.skip_financial:
            print(f"   Financial loss model R²: {results_financial['metrics']['R2']:.4f}")
        return

    # MODE 2: Full training pipeline
    if not validate_paths(RAW_DATA_PATH):
        sys.exit(1)

    cleaned_df = run_data_cleaning(RAW_DATA_PATH, CLEANED_DATA_PATH)
    cleaned_df = run_clustering(cleaned_df)
    X_train, X_test, y_train, y_test = run_data_splitting(cleaned_df, test_size=args.test_size)
    X_train_encoded, X_test_encoded = run_encoding(X_train, X_test)

    if not args.skip_augment:
        X_train_aug, y_train_aug, _ = run_augmentation(X_train_encoded, y_train)
    else:
        X_train_aug, y_train_aug = X_train_encoded, y_train

    if not args.skip_training:
        results = run_model_training(X_train_aug, X_test_encoded, y_train_aug, y_test, save_path=MODEL_SAVE_PATH)
    else:
        results = None

    if not args.skip_financial:
        results_financial = run_financial_loss_training(
            test_size=args.test_size or DEFAULT_TEST_SIZE,
            n_folds=args.n_folds,
            verbose=args.verbose
        )
    else:
        results_financial = None

    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

    if results:
        print(f"\n🎯 Model Results:")
        print(f"   Test Accuracy: {results['accuracy']:.4f}")
        print(f"   CV Mean: {results['cv_mean']:.4f}")
        print(f"   Model saved: {results.get('model_path', MODEL_SAVE_PATH)}")

    if results_financial:
        print(f"\n🎯 Financial Loss Model Results:")
        print(f"   MAE: {results_financial['metrics']['MAE']:,.3f}M$")
        print(f"   R² Score: {results_financial['metrics']['R2']:,.4f}")
        print(f"   Model saved: {results_financial['save_info']['model_path']}")

    print(f"\n💡 Next steps:")
    print(f"   - Use saved model: python main.py --inference")
    print(f"   - Deploy to cloud with: {MODEL_SAVE_PATH}")
    print("\n")


if __name__ == "__main__":
    main()
