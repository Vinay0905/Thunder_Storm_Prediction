# Line-by-Line Technical Breakdown: `preprocess.py`

This document provides a exhaustive explanation for every single line of code in the `preprocess.py` module.

---

### **Header & Imports**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 1-3 | `# Data Match Mystery...` | Comments discussing the column-shift debugging we did. |
| 6-8 | `# In a notebook...` | Warning about "Data Leakage" and why global medians are bad for production. |
| 9-15 | `''' The Pro Way... '''` | High-level strategy: Fix Train, then use those stats for Eval/Holdout. |
| 17-22 | `""" Preprocessing... """` | Official docstring explaining the module's responsibilities. |
| 24 | `import pandas as pd` | Standard tool for handling the DataFrames. |
| 25 | `import numpy as np` | Tool for identifying numbers (`np.number`). |
| 26 | `import numpy as np` | (Repeated line - safe, but redundant). |
| 27 | `import joblib` | **Crucial:** Used to save the "State" (medians) to a file on the disk. |
| 28 | `from pathlib import Path` | Object-oriented paths for split loading. |
| 29 | `from pipelines.config.config import CONFIG` | Custom import to pull split directories from `config.yaml`. |
| 30 | `import logging` | For production-grade activity tracking. |
| 31 | `from utility import setup_logging, ensure_dir` | Imports setup for logs and a helper to create folders safely. |
| 33 | `setup_logging()` | Initializes the logging system. |

---

### **Configuration & Global Setup**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 35 | `INPUT_DIR = Path(CONFIG["data"]["raw_splits_dir"])` | Tells the script to look into `data/raw_splits` (via Config). |
| 36 | `OUTPUT_DIR = Path(CONFIG["data"]["processed_dir"])` | Tells the script to save cleaned files into `data/processed` (via Config). |
| 37 | `MODELS_DIR = Path(CONFIG["model"].get("artifacts_dir", "models"))` | Locates the folder for saving stats. If not in config, it defaults to `"models"`. |

---

### **Function: `run_process`**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 39 | `def run_process(...)` | Defines the cleaning pipeline. Default arguments use our config paths. |
| 40 | `logging.info(...)` | Tracks where we are reading the data from. |
| 44 | `try:` | Starts a safety block in case files are missing. |
| 45-47 | `train_df=pd.read_csv(...)` | Loads the three temporary splits from the disk. |
| 48 | `except FileNotFoundError as e:` | Catches errors if `load.py` wasn't run first. |
| 49-50 | `logging.error(...) / return` | Stops the script and explains the problem clearly. |
| 52-53 | `if train_df.empty / logging.error` | If the file exists but has 0 rows, we can't calculate medians. Stop here. |

---

### **The Imputation Logic (Step 2)**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 57 | `logging.info("Calculating medians...")` | Log progress. |
| 58 | `numeric_col=train_df.select_dtypes(include=[np.number]).columns` | Finds every column that contains numbers. **Skips dates and names automatically.** |
| 59 | `train_medians=train_df[numeric_col].median()` | Calculates the "Middle Value" of every numeric column in the **Training** 1981-2010 data. |
| 62 | `ensure_dir(MODELS_DIR / "medians.joblib")` | Makes sure the `models/` folder exists before saving. |
| 63 | `joblib.dump(train_medians, ...)` | **SAVES the medians.** This is the "Brain" of your preprocessor for later. |
| 64 | `logging.info(...)` | Success log. |
| 66 | `for df in [train_df, eval_df, holdout_df]:` | Loops through all three splits. |
| 67 | `df[numeric_col] = df[numeric_col].fillna(train_medians)` | Replaces every NaN with the training median. **Eval and Holdout are now cleaned using Train's brain.** |

---

### **Saving Clean Data**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 70 | `ensure_dir(output_dir / "cleaning_train.csv")` | Safety check for the `data/processed` folder. |
| 71-73 | `train_df.to_csv(...)` | Saves the "Imputed" dataframes to the processed folder. These are ready for Physics Engineering! |
| 74 | `logging.info(...)` | Final confirmation log. |

---

### **Execution Block**
| Line # | Code | Explanation |
| :--- | :--- | :--- |
| 78-79 | `if __name__ == "__main__": / run_process()` | Direct entry point for running this module as a script. |
