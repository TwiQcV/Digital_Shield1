import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

#load data
csv_path = Path("../Digital_Shield_data/processed/Cleaned_Digital_Shield_Dataset.csv")
assert csv_path.exists(), f"CSV not found at: {csv_path.resolve()}"
df = pd.read_csv(csv_path)

df = df.drop(columns=["severity_peer_label", "severity_peer_cluster"], errors="ignore")

def find_col(df, candidates):
    cols_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in cols_map:
            return cols_map[k]

    def norm(s):
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()
    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]

    raise ValueError(f"column not found: {candidates}\navilable column: {list(df.columns)}")

# finanical loss column
loss_col = find_col(df, [
    "financial loss (in million $)",
    "financial loss in million $",
    "financial_loss_in_million",
    "financial loss"
])

#to be assign to the median for the nan
X = df[[loss_col]].copy()
X = X.fillna(X.median(numeric_only=True))

#scalin using RobustScaler()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

#Kmean clustring buidling
kmeans_peer = KMeans(n_clusters=4, n_init="auto", random_state=42)
peer_clusters = kmeans_peer.fit_predict(X_scaled)


centers_orig = scaler.inverse_transform(kmeans_peer.cluster_centers_).ravel()  # shape (4,)
order = np.argsort(centers_orig)

#map building ----
severity_names = ["Low", "Medium", "High", "Critical"]
idx_to_severity_peer = {int(order[i]): severity_names[i] for i in range(4)}

#cluster number and label ----
df["severity_peer_cluster"] = peer_clusters
df["severity_peer_label"]   = [idx_to_severity_peer[c] for c in peer_clusters]

#save
df[[loss_col, "severity_peer_cluster", "severity_peer_label"]].to_csv(
    "severity_kmeans_peer.csv",
    index=False
)
# result
print("✅ Saved -> severity_kmeans_peer.csv")
print(df["severity_peer_label"].value_counts())
