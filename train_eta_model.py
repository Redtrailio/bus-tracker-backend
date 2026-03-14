#!/usr/bin/env python3
"""
ETA Model Training Script — run weekly via cron.
Usage: python train_eta_model.py [--channel-id N] [--dry-run]
"""
import argparse, json, logging, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd, psycopg2
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST","localhost"), "port": int(os.getenv("DB_PORT","5432")),
    "user": os.getenv("DB_USER","transit_app"), "password": os.getenv("DB_PASSWORD","changeme"),
    "dbname": os.getenv("DB_NAME","transit_db"),
}
MODELS_DIR             = Path("models")
MIN_TRAINING_SAMPLES   = 100
MIN_STOP_PAIRS         = 3
MAX_ACCEPTABLE_MAE_SEC = 120
GBRT_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
    "min_samples_leaf": 5, "subsample": 0.8, "loss": "huber",
    "random_state": 42, "validation_fraction": 0.1,
    "n_iter_no_change": 20, "tol": 1e-4,
}


def load_training_data(channel_id=None, min_samples=MIN_TRAINING_SAMPLES):
    query  = "SELECT start_stop_id,end_stop_id,hour_of_day,day_of_week,travel_time_seconds,segment_distance_m,gps_source,departure_time FROM transit.ml_training_data"
    params = None
    if channel_id is not None:
        query += " WHERE channel_id = %s"
        params = (channel_id,)
    query += " ORDER BY departure_time ASC"
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df   = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except psycopg2.OperationalError as e:
        logger.critical(f"DB connection failed: {e}"); raise SystemExit(1)
    logger.info(f"Loaded {len(df):,} rows.")
    if len(df) < min_samples:
        logger.error(f"Insufficient data: {len(df)} rows."); raise SystemExit(1)
    return df


def engineer_features(df):
    df = df.copy()
    df["hour_sin"]    = np.sin(2*np.pi*df["hour_of_day"]/24.0)
    df["hour_cos"]    = np.cos(2*np.pi*df["hour_of_day"]/24.0)
    df["dow_sin"]     = np.sin(2*np.pi*df["day_of_week"]/7.0)
    df["dow_cos"]     = np.cos(2*np.pi*df["day_of_week"]/7.0)
    df["is_peak_hour"] = (df["hour_of_day"].between(7,9)|df["hour_of_day"].between(16,19)).astype(int)
    if df["segment_distance_m"].isna().any():
        df["segment_distance_m"] = df.groupby(["start_stop_id","end_stop_id"])["segment_distance_m"].transform(lambda x: x.fillna(x.median()))
        df["segment_distance_m"].fillna(df["segment_distance_m"].median(), inplace=True)
    df["travel_time_seconds_log"] = np.log1p(df["travel_time_seconds"])
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),
         ["start_stop_id","end_stop_id"]),
        ("passthrough_numeric","passthrough",
         ["hour_sin","hour_cos","dow_sin","dow_cos","segment_distance_m","is_peak_hour"]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline(steps=[("preprocessor",preprocessor),("model",GradientBoostingRegressor(**GBRT_PARAMS))])


def evaluate_pipeline(pipeline, X_train, X_test, y_train_log, y_test_raw):
    y_pred = np.maximum(np.expm1(pipeline.predict(X_test)), 0)
    mae    = mean_absolute_error(y_test_raw, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2     = r2_score(y_test_raw, y_pred)
    within_60s = np.mean(np.abs(y_test_raw.values - y_pred) <= 60) * 100
    cv_scores  = cross_val_score(pipeline, X_train, y_train_log,
                                  cv=KFold(5,shuffle=True,random_state=42),
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
    logger.info(f"MAE={mae:.1f}s  RMSE={rmse:.1f}s  R²={r2:.4f}  Within60s={within_60s:.1f}%")
    model = pipeline.named_steps["model"]
    try:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = dict(sorted(zip([str(n) for n in names],
            [round(float(i),4) for i in model.feature_importances_]),
            key=lambda x: x[1], reverse=True))
    except Exception:
        importances = {}
    return {"mae_seconds":round(mae,1),"rmse_seconds":round(rmse,1),"r2_score":round(r2,4),
            "within_60s_pct":round(within_60s,1),"cv_mae":round(-cv_scores.mean(),4),
            "n_estimators_used":model.n_estimators_,"feature_importances":importances}


def save_model(pipeline, eval_results, metadata, dry_run=False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version   = len(sorted(MODELS_DIR.glob("eta_model_v*_*.joblib"))) + 1
    model_path  = MODELS_DIR / f"eta_model_v{version}_{timestamp}.joblib"
    report_path = MODELS_DIR / f"training_report_{timestamp}.json"
    latest_path = MODELS_DIR / "eta_model_latest.joblib"
    if dry_run:
        logger.info(f"[DRY RUN] Would save to {model_path}"); return None
    mae = eval_results["mae_seconds"]
    joblib.dump(pipeline, model_path, compress=3)
    promoted = mae <= MAX_ACCEPTABLE_MAE_SEC
    if promoted:
        shutil.copy2(model_path, latest_path)
        logger.info(f"Model saved and promoted to latest.")
    else:
        logger.error(f"MAE {mae}s exceeds gate {MAX_ACCEPTABLE_MAE_SEC}s. NOT promoted.")
    with open(report_path,"w") as f:
        json.dump({"promoted":promoted,"trained_at":datetime.now(timezone.utc).isoformat(),
                   "evaluation":eval_results,"metadata":metadata,"config":GBRT_PARAMS}, f, indent=2)
    return model_path


def train(channel_id=None, min_samples=MIN_TRAINING_SAMPLES, dry_run=False):
    df_raw = load_training_data(channel_id, min_samples)
    df     = engineer_features(df_raw)
    X = df[["start_stop_id","end_stop_id","hour_sin","hour_cos","dow_sin","dow_cos","segment_distance_m","is_peak_hour"]]
    y_log = df["travel_time_seconds_log"]
    y_raw = df["travel_time_seconds"]
    X_train, X_test, y_train_log, y_test_log = train_test_split(X,y_log,test_size=0.2,shuffle=False)
    _,_,y_train_raw,y_test_raw               = train_test_split(X,y_raw,test_size=0.2,shuffle=False)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train_log)
    eval_results = evaluate_pipeline(pipeline, X_train, X_test, y_train_log, y_test_raw)
    metadata = {"total_rows":len(df_raw),"channel_id":channel_id,
                "unique_pairs":int(df_raw.groupby(["start_stop_id","end_stop_id"]).ngroups)}
    save_model(pipeline, eval_results, metadata, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-id", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=MIN_TRAINING_SAMPLES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.channel_id, args.min_samples, args.dry_run)
