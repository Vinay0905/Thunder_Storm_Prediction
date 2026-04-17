"""
Reason: 
    This is the "brain" of your API. 
    It handles Model Loading (using joblib), 
    Imputation (filling missing values with the medians we calculated), 
    and Feature Engineering (creating the complex indices like Environmental Stability that your model expects).
"""

import joblib
 
import pandas as pd 
import numpy as np 
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.FeaturePipeline.FeatureEngineering import apply_feature_engineering
MODEL_PATH = Path("models/xgboost_model.joblib")
MEDIANS_PATH = Path("models/medians.joblib")
FEATURE_NAMES_PATH = Path("models/Feature_names.joblib")
SCALER_PATH = Path("models/scaler.joblib") # <--- NEW


def load_artifacts():
    model=joblib.load(MODEL_PATH)
    medians=joblib.load(MEDIANS_PATH)
    feature_names=joblib.load(FEATURE_NAMES_PATH)
    scaler=joblib.load(SCALER_PATH)

    if 'target' in feature_names :
         feature_names.remove('target')

    return model,medians,feature_names,scaler

def prepare_data_for_Prediction(raw_data:dict,medians:pd.Series,feature_names:List,scaler):
     # 1. Convert to DataFrame
    df = pd.DataFrame([raw_data])
    
    # 2. Apply the SAME logic used in training pipeline
    df = apply_feature_engineering(df)
    
    # 3. Imputation (using the saved medians)
    for col, val in medians.items():
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(val)

    df = df[feature_names]
    
    # 5. NEW: Apply Scaling
    # This turns your 'haywire' numbers into standard units the model understands
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
            
    return df_scaled