# We have a "Data Match" mystery!
# This often happens if one column is an int and the other is a float, or if there's a hidden rounding issue. 
# Before moving to Step 2,you might want to add print(df_indices.dtypes) and print(df_surface.dtypes) in your load.py to ensure Year, Month, and Date are identical types.


# In a notebook, we usually just do df.fillna(df.median()). 
# In production, this is forbidden because it causes "Data Leakage." 
# If you use the median of the entire dataset, your 1981 model "knows" something about the 2020 weather.
'''
The Pro Way:

Calculate (Fit) the median only on the Train set.
Use that same median to fill Eval and Holdout.
Save that median value to a file so that when you predict a single day in the future (Inference), you use the same scaling!
'''

"""
Preprocessing: Imputation & Outlier removal.
- Fits imputer on TRAIN only.
- Applies the same statistics to Eval/Holdout.
- Saves the imputer/statistics for Inference.
"""

import pandas as pd
import numpy as np
import numpy as np 
import joblib 
from pathlib import Path
from pipelines.config.config import CONFIG
import logging 
from utility import setup_logging,ensure_dir

setup_logging()

INPUT_DIR = Path(CONFIG["data"]["raw_splits_dir"])
OUTPUT_DIR = Path(CONFIG["data"]["processed_dir"])
MODELS_DIR = Path(CONFIG["model"].get("artifacts_dir", "models"))

def run_process(input_dir:Path=INPUT_DIR,output_dir:Path=OUTPUT_DIR):
    logging.info(f"Loading splits from {input_dir}...")

    #1. Load Splits

    try:
        train_df=pd.read_csv(input_dir / "train.csv")
        eval_df=pd.read_csv(input_dir / "eval.csv")
        holdout_df=pd.read_csv(input_dir/"holdout.csv")
    except FileNotFoundError as e:
        logging.error(f"Missing split files in {input_dir}. Run load.py first")
        return

    if train_df.empty:
        logging.error("Train dataset is empty. Cannot proceed with imputation.")
        
    #2 Impute Misising Values (Fit on Train, transform all)        

    logging.info("Calculating medians on Train split...")
    numeric_col=train_df.select_dtypes(include=[np.number]).columns
    train_medians=train_df[numeric_col].median()

    # Save statistics for later use in Inference
    ensure_dir(MODELS_DIR / "medians.joblib")
    joblib.dump(train_medians, MODELS_DIR / "medians.joblib")
    logging.info(f"✅ Medians saved to {MODELS_DIR}/medians.joblib")
     # Apply to all
    for df in [train_df, eval_df, holdout_df]:
        df[numeric_col] = df[numeric_col].fillna(train_medians)
    
    # 3. Save
    ensure_dir(output_dir / "cleaning_train.csv")
    train_df.to_csv(output_dir / "cleaning_train.csv", index=False)
    eval_df.to_csv(output_dir / "cleaning_eval.csv", index=False)
    holdout_df.to_csv(output_dir / "cleaning_holdout.csv", index=False)
    logging.info(f"✅ Preprocessing complete. Processed files saved to {output_dir}")



if __name__ == "__main__":
    run_process()

    
