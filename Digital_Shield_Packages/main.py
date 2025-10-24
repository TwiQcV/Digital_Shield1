import sys
import argparse
from pathlib import Path

# Import custom modules
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



def validate_paths(raw_path: str) -> bool:
    """Validate that data paths exist."""
    if not Path(raw_path).exists():
        print(f"❌ Error: Raw data not found at {raw_path}")
        return False
    print(f"✅ Data path verified: {raw_path}")
    return True


def run_data_cleaning(raw_path: str, cleaned_path: str) -> object:
    """Execute data cleaning pipeline."""
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)

    try:
        cleaner = DataCleaner(raw_path)

        # Run full cleaning pipeline
        cleaned_df = cleaner.run_full_pipeline(
            text_cols=TEXT_COLUMNS,
            spelling_maps=SPELLING_MAPS,
            dtype_map=DTYPE_MAP,
            numeric_cols=NUMERIC_COLUMNS,
            categorical_cols=CATEGORICAL_COLUMNS,
            drop_cols=DROP_COLUMNS
        )

        # Validate cleaned dataset
        validation = cleaner.validate_dataset()
        print(f"\n📊 Dataset Validation:")
        print(f"   Total Rows: {validation['total_rows']}")
        print(f"   Total Columns: {validation['total_cols']}")
        print(f"   Null Values: {validation['null_values']}")
        print(f"   Duplicate Rows: {validation['duplicate_rows']}")
        print(f"   Memory Usage: {validation['memory_usage_mb']} MB")

        # Save cleaned data
        Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"\n💾 Cleaned data saved to: {cleaned_path}")


        return cleaned_df

    except Exception as e:
        print(f"❌ Data cleaning failed: {str(e)}")
        sys.exit(1)

def run_clustering(df, feature_column: str = "financial loss (in million $)", severity_path: str = None) -> object:
	"""Execute severity clustering."""

	print("\n" + "="*70)
	print("STEP 1.5: SEVERITY CLUSTERING")
	print("="*70)

	try:
		# Add severity clusters
		df, severity_stats = add_severity_to_dataset(
			df,
			feature_column=feature_column,
			n_clusters=4,
			random_state=42
		)

		# Save with severity column
		if severity_path:
			Path(severity_path).parent.mkdir(parents=True, exist_ok=True)
			df.to_csv(severity_path, index=False)
			print(f"\n💾 Dataset with severity saved to: {severity_path}")

		return df

	except Exception as e:
		print(f"❌ Severity clustering failed: {str(e)}")
		sys.exit(1)


def run_data_splitting(df, target_col: str = "severity", test_size: float = 0.3):
    """Execute data splitting."""
    print("\n" + "="*70)
    print("STEP 2: DATA SPLITTING")
    print("="*70)

    try:
        X_train, X_test, y_train, y_test = split_data(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_state=42
        )

        print(f"\n✅ Data split successfully:")
        print(f"   Training set size: {X_train.shape[0]} rows × {X_train.shape[1]} features")
        print(f"   Test set size: {X_test.shape[0]} rows × {X_test.shape[1]} features")
        print(f"   Target distribution (train):\n{y_train.value_counts()}")
        print(f"\n   Target distribution (test):\n{y_test.value_counts()}")

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
        print(f"   Training set size after encoding: {X_train_encoded.shape}")
        print(f"   Test set size after encoding: {X_test_encoded.shape}")
        print(f"   New feature count: {X_train_encoded.shape[1]}")

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
        print(f"   Augmented training set size: {X_train_aug.shape[0]} rows × {X_train_aug.shape[1]} features")
        print(f"   New target distribution:")
        print(f"{y_train_aug.value_counts()}")

        return X_train_aug, y_train_aug, balanced_df

    except Exception as e:
        print(f"❌ Data augmentation failed: {str(e)}")
        sys.exit(1)


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Digital Shield Cybersecurity Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline with defaults
  python main.py --skip-augment     # Skip augmentation step
  python main.py --test-size 0.2    # Use 20% test split
        """
    )

    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip SMOTE augmentation step"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set proportion (default: 0.3)"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="severity",
        help="Target column name (default: 'severity')"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🛡️  DIGITAL SHIELD CYBERSECURITY DATA PIPELINE")
    print("="*70)
    print(f"\n📅 Configuration:")
    print(f"   Raw data path: {RAW_DATA_PATH}")
    print(f"   Cleaned data path: {CLEANED_DATA_PATH}")
    print(f"   Test size: {args.test_size * 100}%")
    print(f"   Target column: {args.target_col}")
    print(f"   Skip augmentation: {args.skip_augment}")

    # Step 1: Validate paths
    if not validate_paths(RAW_DATA_PATH):
        sys.exit(1)

    # Step 2: Data cleaning
    cleaned_df = run_data_cleaning(RAW_DATA_PATH, CLEANED_DATA_PATH)

    # Step 1.5: Severity clustering
    severity_path = CLEANED_DATA_PATH.replace(".csv", "_with_severity.csv")
    cleaned_df = run_clustering(cleaned_df, severity_path=severity_path)

    # Step 3: Data splitting
    X_train, X_test, y_train, y_test = run_data_splitting(
        cleaned_df,
        target_col="severity_kmeans",
        test_size=args.test_size
    )

    # Step 4: Feature encoding
    X_train_encoded, X_test_encoded = run_encoding(X_train, X_test)

    # Step 5: Data augmentation (optional)
    if not args.skip_augment:
        X_train_aug, y_train_aug, balanced_df = run_augmentation(X_train_encoded, y_train)
        print(f"\n💾 Balanced dataset saved as 'balanced_df' variable")
    else:
        print("\n⏭️  Augmentation step skipped")
        X_train_aug, y_train_aug = X_train_encoded, y_train

    # Final summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n📦 Output Summary:")
    print(f"   X_train shape: {X_train_aug.shape}")
    print(f"   X_test shape: {X_test_encoded.shape}")
    print(f"   y_train shape: {y_train_aug.shape}")
    print(f"   y_test shape: {y_test.shape}")
    print(f"\n💡 Next steps:")
    print(f"   - Train ML models on (X_train_aug, y_train_aug)")
    print(f"   - Evaluate on (X_test_encoded, y_test)")
    print(f"   - Adjust hyperparameters as needed")
    print("\n")

if __name__ == "__main__":
    main()
