"""
Hyperparameter tuning with Optuna + MLflow.

- Optimizes XGB params on eval set RMSE.
- Logs trials to MLflow.
- Retrains best model and saves to `model_output`.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import optuna
import mlflow
import mlflow.xgboost
from pathlib import Path
from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Path setup
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def load_config(config_path: str = "config.yaml"):
    with open(PROJECT_ROOT / config_path, 'r') as f:
        return yaml.safe_load(f)

def _load_data(config):
    df = pd.read_csv(PROJECT_ROOT / config['data']['processed_final'])
    target = config['features']['target_col']
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def tune_model(
    config_path: str = "config.yaml",
    n_trials: int = 15,
    experiment_name: str = "thunderstorm_tuning"
):
    config = load_config(config_path)
    mlflow.set_experiment(experiment_name)
    
    X, y = _load_data(config)
    random_state = config['model']['random_state']

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }

        with mlflow.start_run(nested=True):
            model = XGBClassifier(**params)
            model.fit(X, y)
            
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            
            # We'll optimize for log_loss (minimize) or f1_score (maximize)
            current_logloss = log_loss(y, y_proba)
            current_acc = accuracy_score(y, y_pred)
            
            mlflow.log_params(params)
            mlflow.log_metrics({"logloss": current_logloss, "accuracy": current_acc})

        return current_logloss

    # 1. Start Tuning
    print(f"🎯 Starting {n_trials} trials of hyperparameter tuning...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best params discovered:", study.best_params)

    # 2. Retrain Best Model
    best_params = study.best_params
    best_model = XGBClassifier(**{**best_params, "random_state": random_state})
    best_model.fit(X, y)

    # 3. Save Final Artifacts
    out_path = PROJECT_ROOT / config['model']['path']
    joblib.dump(best_model, out_path)
    joblib.dump(list(X.columns), out_path.parent / "Feature_names.joblib")
    
    # 4. Log Final Results to MLflow
    with mlflow.start_run(run_name="best_tuned_model"):
        mlflow.log_params(best_params)
        mlflow.xgboost.log_model(best_model, "model")

    print(f"🚀 Best model saved and logged to MLflow!")
    return best_params

if __name__ == "__main__":
    tune_model()
