import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from xgboost import DMatrix, train as xgb_train

csv = next((Path(p) for p in ["./Data Augmentetion.csv","./Data Augmentation.csv"] if Path(p).exists()), None)
assert csv is not None, "CSV file not found next to notebook."
df = pd.read_csv(csv)

# 2) target selection
target = next(t for t in ["Financial Loss (in Million $)", "financial loss (in million $)"] if t in df.columns)
X = df.drop(columns=[target]).copy()
y = pd.to_numeric(df[target], errors="coerce").copy()

# drop the 'NaN' target rows
mask_ok = y.notna()
X, y = X.loc[mask_ok].reset_index(drop=True), y.loc[mask_ok].reset_index(drop=True)


if "year" in X.columns:
    X["year"] = pd.to_numeric(X["year"], errors="coerce") #the column year to be numiric

for c in ["number of affected users", "data breach in gb", "incident resolution time (in hours)"]: # no -values
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower=0)

if (y < 0).any():
    raise ValueError("Negative values in target — check data before using log1p") # to make sure no -values as we are using log


# 4) Feature engineering
def add_if(D, cols, name, f):
    if all(c in D.columns for c in cols) and name not in D.columns:
        try:
            D[name] = f(*(D[c].fillna(0) for c in cols))
            D[name] = pd.to_numeric(D[name], errors="coerce")
        except Exception:
            D[name] = np.nan

add_if(X, ["number of affected users"], "log_users", lambda u: np.log1p(u))
add_if(X, ["data breach in gb"], "log_breach", lambda g: np.log1p(g))
add_if(X, ["incident resolution time (in hours)"], "log_resolution_time", lambda t: np.log1p(t))
add_if(X, ["number of affected users","data breach in gb"], "impact_index", lambda u,g: u*np.log1p(g))
add_if(X, ["number of affected users","incident resolution time (in hours)"], "users_per_hour", lambda u,t: u/(1.0+t))
add_if(X, ["year"], "years_since_2010", lambda yr: pd.to_numeric(yr, errors="coerce") - 2010)

if {"impact_index","users_per_hour"} <= set(X.columns):
    X["severity_ratio"] = X["impact_index"] / (1.0 + X["users_per_hour"]) #severity_ratio > to mesure the severity of the incident compared to how fast it spreads.
if {"log_users","log_breach"} <= set(X.columns): # complexity > Shows the scale of the incident based on the number of people affected and the size of the data leak.
    X["complexity"] = X["log_users"] * X["log_breach"]

X.replace([np.inf, -np.inf], np.nan, inplace=True) # remove infinities because we used log


# 5) Split Train/Test
X_tr_all, X_te_all, y_tr_all, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)


# 6) KFold setup (no leakage version)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_te_all))
best_iters = []


# 7) Loop folds — fit preprocessing inside each fold
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_all), 1):
    X_tr, X_va = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]     #we did the train and validate all within the training data not the test set prevents data leakage.
    y_tr, y_va = y_tr_all.iloc[tr_idx], y_tr_all.iloc[va_idx]

    # detect numeric & categorical and reassign them to make suere in the training phase
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_tr.select_dtypes(include=["object","category","bool"]).columns.tolist() #> Important because after FE, we now have additional numerical features.

    # preprocessing (fit per fold)
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

    # fit/transform inside fold
    prep = preprocess.fit(X_tr)
    X_tr_p = prep.transform(X_tr)
    X_va_p = prep.transform(X_va)
    X_te_p = prep.transform(X_te_all)

    # get feature names
    try:
        ohe_step = prep.named_transformers_["cat"].named_steps["enc"]
        ohe_names = ohe_step.get_feature_names_out(cat_cols).tolist()
    except Exception:
        ohe_names = []
    feature_names = num_cols + ohe_names

    # log-transform for the target
    y_tr_log, y_va_log = np.log1p(y_tr.values), np.log1p(y_va.values)

    # DMatrix is XGBoost’s custom data structure designed for efficiency and speed.
    dtr = DMatrix(X_tr_p, label=y_tr_log, feature_names=feature_names) #for training
    dva = DMatrix(X_va_p, label=y_va_log, feature_names=feature_names) #for validation
    dte = DMatrix(X_te_p, feature_names=feature_names) # for testing

    # XGBoost params
    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse","mae"],
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

    #The holdout prevents overfitting by stopping at the point where the model has truly learned best
    bst = xgb_train(
        params=params,
        dtrain=dtr,
        num_boost_round=3000,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    best_it = int(bst.best_iteration) if bst.best_iteration else 2999
    best_iters.append(best_it + 1)

    pred_log = bst.predict(dte, iteration_range=(0, best_it+1))
    fold_preds = np.expm1(pred_log)
    fold_preds = np.clip(fold_preds, 0, float(y_tr_all.max()) * 10)
    test_preds += fold_preds / kf.get_n_splits()


# 8) Metrics (no leakage)
MAE = mean_absolute_error(y_te, test_preds)
RMSE = np.sqrt(mean_squared_error(y_te, test_preds))
R2 = r2_score(y_te, test_preds)
MedianAE = median_absolute_error(y_te, test_preds)
sMAPE = 100 * np.mean(2 * np.abs(y_te - test_preds) / (np.abs(y_te) + np.abs(test_preds) + 1e-8))

print("✅ XGBoost — no data leakage version")
print("Best trees per fold:", best_iters, " | avg:", int(np.mean(best_iters)))
print("-"*74)
print("📊 Test (Holdout) Metrics")
print(f"{'MAE':<20}: {MAE:,.3f}")
print(f"{'Median AE':<20}: {MedianAE:,.3f}")
print(f"{'RMSE':<20}: {RMSE:,.3f}")
print(f"{'R²':<20}: {R2:,.4f}")
print(f"{'sMAPE':<20}: {sMAPE:,.2f}%")


err = np.abs(y_te.values - test_preds)
bad_idx = np.argsort(-err)[:10]
diagnostic = pd.DataFrame({
    "y_true": y_te.values[bad_idx],
    "y_pred": test_preds[bad_idx],
    "abs_err": err[bad_idx]
})
print("\nTop 10 worst absolute errors on holdout:")
print(diagnostic.to_string(index=False))
