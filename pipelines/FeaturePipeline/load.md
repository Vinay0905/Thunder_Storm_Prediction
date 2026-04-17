# Line-by-Line Technical Breakdown: `load.py`

This document provides a exhaustive explanation for every single line of code in the `load.py` module.

---

### **Header & Imports**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 1 | `#Leakage Prevention:` | A comment explaining the core goal: avoiding "Data Leakage". |
| 2 | `# We split by Year before doing anything else.` | Explaining the strategy: using time-series splitting. |
| 3 | `# This ensures...` | Rationale: past data shouldn't know about future data. |
| 4 | `import logging` | Imports Python's built-in logging module for production monitoring. |
| 5 | `import pandas as pd` | The primary library for data manipulation (DataFrames). |
| 6 | `from pathlib import Path` | Object-oriented filesystem paths (better than string manipulation). |
| 7 | `from pipelines.config.config import CONFIG` | Custom import to pull settings from your `config.yaml` file. |
| 8 | `import numpy as np` | Library for numerical operations and handling NaNs (Not-a-Number). |
| 9 | `from utility import setup_logging` | Pulls your helper function to standardize how logs look. |
| 10 | ` ` | Empty line for readability (PEP 8 standard). |
| 11 | `setup_logging()` | Immediately calls the logger setup so imports/actions are tracked. |
| 12-16 | `...comments...` | Notes on testing and readability. |
| 17 | `DATA_RAW_DIR = Path("data/raw_splits")` | A global constant defining where the merged splits will live. |
| 19-25 | `...comments...` | Overview of the steps inside the function. |

---

### **Function Definition: `load_and_Split_data`**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 27 | `def load_and_Split_data(output_dir : Path \| str = DATA_RAW_DIR,):` | Defines the main function. It takes an optional `output_dir` which defaults to our global constant. |
| 28 | `logging.info("Starting Data Load and Split...")` | Logs the start of the process (visible in your terminal/log file). |
| 29 | `# 1. Load using paths...` | Comment marking Step 1. |
| 30 | ` ` | Empty line. |
| 31 | `df_indices=pd.read_csv(CONFIG['data']['raw_indices'],index_col=False)` | Reads the Upper-Air data. **`index_col=False`** is the "Magic Fix" that prevents Python from shifting columns due to trailing commas. |
| 32 | `df_surface=pd.read_csv(CONFIG['data']['raw_surfaces'],index_col=False)` | Reads the Ground Surface data, using the path defined in your `config.yaml`. |
| 33 | `# Add this diagnostic...` | Comment. |
| 34 | `logging.info(f"GMT Unique Values: {df_indices['GMT'].unique()}")` | Diagnostic log to see what values are actually in the GMT column. |
| 35 | `logging.info(f"GMT Dtype: {df_indices['GMT'].dtype}")` | Diagnostic log to check if GMT is an integer, float, or object. |
| 36 | ` ` | Empty line. |
| 37 | `# 2. Initial Cleaning...` | Comment marking Step 2. |
| 38 | `df_surface = df_surface.replace(r'\s*$', np.nan, regex=True)` | Finds any cell that is empty or just whitespace and turns it into `NaN`. |
| 39 | `df_surface = df_surface.apply(pd.to_numeric, errors='coerce')` | Forces the entire dataframe to be numbers. If something isn't a number, it becomes `NaN`. |
| 40 | `df_surface = df_surface.rename(columns={'YEAR': 'Year', 'MN': 'Month', 'DT': 'Date'})` | Uniforms naming so we can merge with the Indices file later. |
| 41 | ` ` | Empty line. |
| 42 | `# Create target...` | Comment. |
| 43 | ` ` | Empty line. |
| 44 | `df_surface['target']=((df_surface['TH']==1) \| (df_surface['HA']==1)).astype(int)` | Creates the "Answer" column: 1 if Hail or Thunderstorm happened, else 0. |
| 45 | `# 3. Merge Indices...` | Comment marking Step 3. |
| 46 | `df_indices=df_indices[df_indices['GMT']==0].copy()` | Filters the Indices to only include 0 GMT launches (Meteorological setup). |
| 47 | `df=pd.merge(` | Starts the merging process. |
| 48 | `df_indices,df_surface[` | First DF is Indices, second is a subset of Surface. |
| 49 | `['Year','Month','Date','target']],` | We only take the date columns and our new target from Surface. |
| 50 | `on=['Year','Month','Date'],` | The shared "Keys" that define a unique row. |
| 51 | `how='left'` | Keeps all data from the left (Indices) even if no ground data was recorded. |
| 52 | `)` | Closes the merge function. |

---

### **Splitting & Saving**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 54 | `#4 .TIme Based Splitting` | Comment. |
| 55 | `train_df=df[df['Year']<=2010]` | Training set: All data from 1981 to 2010. |
| 56 | `eval_df=df[(df['Year']>2010) & (df['Year']<=2015)]` | Validation set: 2011 to 2015. |
| 57 | `holdout_df=df[df['Year'] >2015]` | Final Test set: 2016 to 2020. |
| 58 | `#5 Save Splits` | Comment. |
| 59 | ` ` | Empty line. |
| 60 | `output_dir=Path(output_dir)` | Ensures the output directory is a `Path` object (safety). |
| 61 | `output_dir.mkdir(parents=True,exist_ok=True)` | Automatically creates the folders (`data/raw_splits`) if they don't exist. |
| 62 | ` ` | Empty line. |
| 63 | `train_df.to_csv(output_dir / "train.csv",index=False)` | Saves the Training split. `index=False` prevents an extra boring column. |
| 64 | `eval_df.to_csv(output_dir / "eval.csv",index=False)` | Saves the Evaluation split. |
| 65 | `holdout_df.to_csv(output_dir / "holdout.csv",index=False)` | Saves the Holdout split. |
| 66 | ` ` | Empty line. |
| 67 | `logging.info(f"✅ Data split completed. Saved to {output_dir}")` | Confirms success in the logs. |
| 68 | `print(f" Train: {train_df.shape}...` | Prints a summary to your screen so you can see the row counts. |
| 69 | `return train_df, eval_df, holdout_df` | Hands the data back to any other Python script that calls this. |

---

### **Execution Block**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 73 | `if __name__=="__main__":` | Only run the code below if this script is executed directly (not imported). |
| 74 | `load_and_Split_data()` | Triggers the entire process. |
