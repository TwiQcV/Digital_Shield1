import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict


class SeverityClusterer:
    """
    K-Means clustering for severity classification.
    Maps financial loss to severity levels: Low, Medium, High, Critical.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """Initialize clusterer with hyperparameters."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.kmeans = None
        self.severity_map = None
        self.clustering_log = []

    def fit(self, X: np.ndarray) -> 'SeverityClusterer':
        """Fit the clustering model to data."""
        print("[Clustering] Fitting K-Means model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = self.kmeans.fit_predict(X_scaled)

        # Map clusters to severity by impact score
        centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        impact = centers.sum(axis=1)
        order = np.argsort(impact)
        self.severity_map = {order[i]: ["Low", "Medium", "High", "Critical"][i]
                            for i in range(self.n_clusters)}

        self.clustering_log.append({
            'step': 'Fit K-Means',
            'n_clusters': self.n_clusters,
            'samples': X.shape[0],
            'features': X.shape[1],
            'status': 'SUCCESS'
        })

        print(f" ✓ Fitted K-Means with {self.n_clusters} clusters")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict severity for new data."""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        return np.array([self.severity_map[c] for c in clusters])

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one call."""
        self.fit(X)
        return self.predict(X)

    def get_clustering_report(self) -> pd.DataFrame:
        """Return detailed clustering report."""
        return pd.DataFrame(self.clustering_log)


def add_severity_to_dataset(df: pd.DataFrame,
                            feature_column: str,
                            n_clusters: int = 4,
                            random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Add severity column to dataframe using K-Means clustering.

    Args:
        df: Input dataframe
        feature_column: Column name to use for clustering
        n_clusters: Number of severity clusters
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with severity column, clustering statistics)
    """
    print("[Step 1.5] ADDING SEVERITY CLUSTERS")
    print("="*70)

    try:
        # Extract features
        X = df[[feature_column]].values

        # Initialize and fit clusterer
        clusterer = SeverityClusterer(n_clusters=n_clusters, random_state=random_state)
        severity_labels = clusterer.fit_predict(X)

        # Add severity column
        df["severity_kmeans"] = severity_labels

        # Calculate statistics
        severity_stats = {
            'total_samples': len(df),
            'severity_distribution': df["severity_kmeans"].value_counts().to_dict(),
            'clustering_feature': feature_column,
            'status': 'SUCCESS'
        }

        print(f"\n✅ Severity clusters added successfully:")
        print(f" Clustering feature: {feature_column}")
        print(f" Severity distribution:")
        for severity, count in severity_stats['severity_distribution'].items():
            percentage = (count / len(df)) * 100
            print(f"  - {severity}: {count} samples ({percentage:.1f}%)")

        return df, severity_stats

    except Exception as e:
        print(f"❌ Severity clustering failed: {str(e)}")
        raise
