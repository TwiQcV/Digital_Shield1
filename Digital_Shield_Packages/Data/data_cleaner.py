import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from typing import Dict, List

class DataCleaner:


    def __init__(self, filepath: str):
        """Initialize cleaner with raw dataset filepath."""
        self.filepath = filepath
        self.df_original = None
        self.df_current = None
        self.cleaning_log = []

    def load_dataset(self) -> pd.DataFrame:
        """Load CSV and normalize column names."""
        print("[Step 1/6] Loading dataset...")
        self.df_original = pd.read_csv(self.filepath)
        self.df_current = self.df_original.copy()

        # Normalize column names
        self.df_current.columns = self.df_current.columns.str.strip().str.lower()

        initial_shape = self.df_current.shape
        self.cleaning_log.append({
            'step': 'Load Dataset',
            'rows': initial_shape[0],
            'cols': initial_shape[1],
            'status': 'SUCCESS'
        })
        print(f"  ✓ Loaded {initial_shape[0]} rows, {initial_shape[1]} columns")
        return self.df_current

    @staticmethod
    def normalize_text(s: str) -> str:
        """Normalize text: lowercase, remove special chars, trim whitespace."""
        if pd.isna(s):
            return np.nan
        s = str(s).strip().lower()
        s = re.sub(r'[^a-z0-9]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s if s else np.nan

    def normalize_text_columns(self, cols: List[str]) -> pd.DataFrame:
        """Apply text normalization to specified columns."""
        print("[Step 2/6] Normalizing text columns...")
        before_state = self.df_current[cols].copy()

        for col in cols:
            self.df_current[col] = (
                self.df_current[col].astype(str)
                .apply(self.normalize_text)
                .replace({'nan': np.nan})
            )

        # Validation: check for data loss
        nulls_before = before_state.isna().sum().sum()
        nulls_after = self.df_current[cols].isna().sum().sum()

        self.cleaning_log.append({
            'step': 'Normalize Text',
            'columns_affected': len(cols),
            'nulls_before': int(nulls_before),
            'nulls_after': int(nulls_after),
            'status': 'SUCCESS'
        })
        print(f"  ✓ Normalized {len(cols)} text columns")
        return self.df_current

    def unify_spellings(self, col: str, mapping: Dict[str, str]) -> pd.DataFrame:
        """Unify common misspellings using provided mapping."""
        print(f"[Step 3/6] Unifying spellings in '{col}'...")
        before_unique = self.df_current[col].nunique()

        self.df_current[col] = (
            self.df_current[col]
            .map(mapping)
            .fillna(self.df_current[col])
        )

        after_unique = self.df_current[col].nunique()

        self.cleaning_log.append({
            'step': f'Unify Spellings ({col})',
            'unique_before': before_unique,
            'unique_after': after_unique,
            'mappings_applied': len(mapping),
            'status': 'SUCCESS'
        })
        print(f"  ✓ Unified spellings ({before_unique} → {after_unique} unique values)")
        return self.df_current

    def convert_types(self, dtype_map: Dict[str, str]) -> pd.DataFrame:
        """Convert columns to specified data types."""
        print("[Step 4/6] Converting data types...")
        conversion_results = []

        for col, dt in dtype_map.items():
            before_type = str(self.df_current[col].dtype)
            self.df_current[col] = pd.to_numeric(
                self.df_current[col],
                errors='coerce'
            ).astype(dt)
            after_type = str(self.df_current[col].dtype)

            conversion_results.append({
                'column': col,
                'from': before_type,
                'to': after_type
            })

        self.cleaning_log.append({
            'step': 'Convert Types',
            'conversions': len(dtype_map),
            'status': 'SUCCESS'
        })
        print(f"  ✓ Converted {len(dtype_map)} columns to target types")
        return self.df_current

    def handle_missing_values(self,
                             num_cols: List[str],
                             cat_cols: List[str],
                             num_strategy: str = 'median',
                             cat_strategy: str = 'most_frequent') -> pd.DataFrame:
        """Impute missing values using sklearn SimpleImputer."""
        print("[Step 5/6] Handling missing values...")

        # Track missing before imputation
        num_missing_before = self.df_current[num_cols].isna().sum().sum()
        cat_missing_before = self.df_current[cat_cols].isna().sum().sum()

        # Impute numeric columns
        if num_cols:
            num_imp = SimpleImputer(strategy=num_strategy)
            self.df_current[num_cols] = num_imp.fit_transform(self.df_current[num_cols])

        # Impute categorical columns
        if cat_cols:
            cat_imp = SimpleImputer(strategy=cat_strategy)
            self.df_current[cat_cols] = cat_imp.fit_transform(self.df_current[cat_cols])

        # Track missing after imputation
        num_missing_after = self.df_current[num_cols].isna().sum().sum()
        cat_missing_after = self.df_current[cat_cols].isna().sum().sum()

        self.cleaning_log.append({
            'step': 'Handle Missing Values',
            'numeric_missing_before': int(num_missing_before),
            'numeric_missing_after': int(num_missing_after),
            'categorical_missing_before': int(cat_missing_before),
            'categorical_missing_after': int(cat_missing_after),
            'status': 'SUCCESS'
        })
        print(f"  ✓ Imputed {int(num_missing_before + cat_missing_before)} missing values")
        return self.df_current

    def drop_irrelevant_columns(self, cols: List[str]) -> pd.DataFrame:
        """Remove specified columns if they exist."""
        print("[Step 6/6] Dropping irrelevant columns...")
        before_cols = self.df_current.shape[1]

        cols_dropped = [c for c in cols if c in self.df_current.columns]
        self.df_current = self.df_current.drop(columns=cols_dropped, errors='ignore')

        after_cols = self.df_current.shape[1]

        self.cleaning_log.append({
            'step': 'Drop Columns',
            'columns_dropped': len(cols_dropped),
            'cols_before': before_cols,
            'cols_after': after_cols,
            'status': 'SUCCESS'
        })
        print(f"  ✓ Dropped {len(cols_dropped)} columns ({before_cols} → {after_cols})")
        return self.df_current

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove exact duplicate rows."""
        print("[Post-Processing] Removing duplicates...")
        before_rows = self.df_current.shape[0]
        self.df_current = self.df_current.drop_duplicates()
        after_rows = self.df_current.shape[0]

        self.cleaning_log.append({
            'step': 'Remove Duplicates',
            'rows_before': before_rows,
            'rows_after': after_rows,
            'duplicates_removed': before_rows - after_rows,
            'status': 'SUCCESS'
        })
        print(f"  ✓ Removed {before_rows - after_rows} duplicate rows")
        return self.df_current

    def finalize_year_column(self) -> pd.DataFrame:
        """Ensure year column is integer type."""
        print("[Post-Processing] Finalizing year column...")
        self.df_current['year'] = self.df_current['year'].astype(int)

        self.cleaning_log.append({
            'step': 'Finalize Year',
            'year_type': str(self.df_current['year'].dtype),
            'status': 'SUCCESS'
        })
        print(f"  ✓ Year column set to int type")
        return self.df_current

    def run_full_pipeline(self,
                         text_cols: List[str],
                         spelling_maps: Dict[str, Dict[str, str]],
                         dtype_map: Dict[str, str],
                         numeric_cols: List[str],
                         categorical_cols: List[str],
                         drop_cols: List[str]) -> pd.DataFrame:
        """Execute complete cleaning pipeline."""
        print("\n" + "="*60)
        print("DIGITAL SHIELD DATA CLEANING PIPELINE")
        print("="*60 + "\n")

        # Step 1: Load
        self.load_dataset()

        # Step 2: Normalize text
        self.normalize_text_columns(text_cols)

        # Step 3: Unify spellings
        for col, mapping in spelling_maps.items():
            self.unify_spellings(col, mapping)

        # Step 4: Convert types
        self.convert_types(dtype_map)

        # Step 5: Handle missing values
        self.handle_missing_values(numeric_cols, categorical_cols)

        # Step 6: Drop columns
        self.drop_irrelevant_columns(drop_cols)

        # Post-processing
        self.remove_duplicates()
        self.finalize_year_column()

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")

        return self.df_current

    def get_cleaning_report(self) -> pd.DataFrame:
        """Return detailed cleaning report."""
        return pd.DataFrame(self.cleaning_log)

    def validate_dataset(self) -> Dict:
        """Validate cleaned dataset integrity."""
        validation = {
            'total_rows': self.df_current.shape[0],
            'total_cols': self.df_current.shape[1],
            'null_values': int(self.df_current.isna().sum().sum()),
            'duplicate_rows': int(self.df_current.duplicated().sum()),
            'memory_usage_mb': round(self.df_current.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        return validation

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return cleaned dataset."""
        return self.df_current
