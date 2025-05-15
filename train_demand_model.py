import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG_FILE = "demand_prediction_log.csv"
MODEL_FILE = "rf_demand_model.pkl"
RETRAIN_MARKER_FILE = "last_retrain_marker.txt"
METRIC_LOG_FILE = "retrain_metrics_log.csv"


def load_verified_data():
    df = pd.read_csv(LOG_FILE)
    return df.dropna(subset=["ActualDemand"])


def get_last_retrain_count():
    if os.path.exists(RETRAIN_MARKER_FILE):
        with open(RETRAIN_MARKER_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_retrain_marker(count):
    with open(RETRAIN_MARKER_FILE, "w") as f:
        f.write(str(count))


def log_metrics(entry_count, avg_rmse, scores):
    metrics = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(),
                "entry_count": entry_count,
                "avg_rmse": avg_rmse,
                "cv_scores": ";".join(f"{-s:.2f}" for s in scores),
            }
        ]
    )

    if os.path.exists(METRIC_LOG_FILE):
        old_metrics = pd.read_csv(METRIC_LOG_FILE)
        metrics = pd.concat([old_metrics, metrics], ignore_index=True)

    metrics.to_csv(METRIC_LOG_FILE, index=False)


def train_model(X, y):
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("reg", RandomForestRegressor(random_state=42))]
    )

    param_grid = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [None, 10, 20],
        "reg__min_samples_split": [2, 5],
    }

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipe, param_grid, cv=inner_cv, scoring="neg_root_mean_squared_error"
    )

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        grid_search, X, y, cv=outer_cv, scoring="neg_root_mean_squared_error"
    )

    avg_rmse = -scores.mean()
    grid_search.fit(X, y)
    joblib.dump(grid_search.best_estimator_, MODEL_FILE)

    return avg_rmse, scores


def retrain():
    df = load_verified_data()
    current_count = len(df)
    last_retrain_count = get_last_retrain_count()

    new_entries = current_count - last_retrain_count
    X = df[["Holiday", "Temperature", "Rainfall", "Condition"]]
    y = df["ActualDemand"]

    avg_rmse, scores = train_model(X, y)
    save_retrain_marker(current_count)
    log_metrics(current_count, avg_rmse, scores)
    return True, f"Retrained on {new_entries} entries. Avg RMSE: {avg_rmse:.2f}"


if __name__ == "__main__":
    retrain()
