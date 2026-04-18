"""
Train a baseline XGBoost model.

- Reads feature-engineered train/eval CSVs.
- Trains XGBRegressor.
- Returns metrics and saves model to `model_output`.
"""

import sys
import pandas as pd 
import numpy as np 
import joblib 
from pathlib import Path 
import yaml
from typing import Dict,Optional
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
PROJECT_ROOT=Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

def load_config(config_path:str="config.yaml"):
    with open(PROJECT_ROOT/config_path,'r') as f:
        return yaml.safe_load(f)
def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
def train_model(
    config_path: str = "config.yaml",
    model_params: Optional[Dict] = None,
    sample_frac: Optional[float] = None
):
    """
    Trains the XGBoost Classifier and saves the model artifact.
    """
    config = load_config(config_path)
    
    # 1. Load Data (Using paths from config.yaml)
    train_df = pd.read_csv(PROJECT_ROOT / config['data']['processed_final'])
    
    # Optional sampling (like in your example)
    train_df = _maybe_sample(train_df, sample_frac, config['model']['random_state'])
    
    # 2. Separate Features and Target
    target = config['features']['target_col']
    X = train_df.drop(columns=[target])
    y = train_df[target]
    
    # 3. Model Parameters
    # Using defaults + any overrides
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": config['model']['random_state'],
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    if model_params:
        params.update(model_params)
    # 4. Train
    print(f"🚀 Training model on {len(X)} rows...")
    model = XGBClassifier(**params)
    model.fit(X, y)
    # 5. Save Model
    out_path = PROJECT_ROOT / config['model']['path']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    
    # Also save feature names (crucial for inference alignment!)
    joblib.dump(list(X.columns), out_path.parent / "Feature_names.joblib")
    
    print(f"✅ Model and Feature Names saved to {out_path.parent}")
    return model
if __name__ == "__main__":
    train_model()
