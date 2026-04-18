"""
Evaluate a saved XGBoost model on the eval split.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Path setup
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def load_config(config_path: str = "config.yaml"):
    with open(PROJECT_ROOT / config_path, 'r') as f:
        return yaml.safe_load(f)

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def evaluate_model(
    config_path: str = "config.yaml",
    sample_frac: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluates the saved XGBoost model on the processed final dataset.
    Note: For a production pipeline, you would typically use a separate 
    test/eval CSV, but here we'll use the final processed data.
    """
    config = load_config(config_path)
    
    # 1. Load Model and Data
    model_path = PROJECT_ROOT / config['model']['path']
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}. Run train.py first!")
        return {}

    eval_df = pd.read_csv(PROJECT_ROOT / config['data']['processed_final'])
    eval_df = _maybe_sample(eval_df, sample_frac, config['model']['random_state'])

    # 2. Prepare Features and Target
    target = config['features']['target_col']
    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    # 3. Load Model and Predict
    model = joblib.load(model_path)
    y_pred = model.predict(X_eval)

    # 4. Calculate Metrics (Classification specific)
    accuracy = float(accuracy_score(y_eval, y_pred))
    report = classification_report(y_eval, y_pred, output_dict=True)
    
    # Extract f1-score for the positive class (assuming it's '1')
    # If your target class is different, adjust '1.0' or '1' below
    pos_label = '1' if '1' in report else '1.0' if '1.0' in report else list(report.keys())[0]
    f1 = report[pos_label]['f1-score']

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1
    }

    print("📊 Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_eval, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_eval, y_pred))
    
    return metrics

if __name__ == "__main__":
    evaluate_model()
