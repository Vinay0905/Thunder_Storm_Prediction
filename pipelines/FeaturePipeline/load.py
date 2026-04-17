#Leakage Prevention: 
#   We split by Year before doing anything else. 
#   This ensures that information from 2020 doesn't "leak" into the training set from 1980.
import logging 
import pandas as pd 
from pathlib import Path
from config import CONFIG
import numpy as np
from utility import setup_logging

setup_logging()

DATA_RAW_DIR = Path("data/raw_splits") # We will save splits here

# 1. Load using paths from YOUR config.yaml
# 2. Initial Cleaning (The 'Column Shift' fix)
# 3. Create target in surface data
# 4. Merge Indices (0 GMT setup) with Ground Outcome (Target)
# 5. Time Based Splitting
# 6. Save Splits


def load_and_Split_data(output_dir : Path | str = DATA_RAW_DIR,):
        logging.info("Starting Data Load and Split...")
        # 1. Load using paths from YOUR config.yaml

        df_indices=pd.read_csv(CONFIG['data']['raw_indices'])
        df_surfaces=pd.read_csv(CONFIG['data']['raw_surfaces'])
        # 2. Initial Cleaning (The 'Column Shift' fix)
        df_surface = df_surface.replace(r'\s*$', np.nan, regex=True)
        df_surface = df_surface.apply(pd.to_numeric, errors='coerce')
        df_surface = df_surface.rename(columns={'YEAR': 'Year', 'MN': 'Month', 'DT': 'Date'})
        
        # Create target in surface data

        df_surface['target']=((df_surface['TH']==1) | (df_surface['HA']==1)).astype(int)
        # 3. Merge Indices (0 GMT setup) with Ground Outcome (Target)
        df_indices=df_indices[df_indices['GMT']==0].copy()
        df=pd.merge(    
            df_indices,df_surface[
                ['Year','Month','Date','target']],
                on=['Year','Month','Date'],
                how='left'
            )

        #4 .TIme Based Splitting
        train_df=df[df['Year']<=2010]
        eval_df=df[(df['Year']>2010) & (df['Year']<=2015)]
        holdout_df=df[df['Year'] >2015]
        #5 Save Splits

        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)

        train_df.to_csv(output_dir / "train.csv",index=False)
        eval_df.to_csv(output_dir / "eval.csv",index=False)
        holdout_df.to_csv(output_dir / "holdout.csv",index=False)

        logging.info(f"✅ Data split completed. Saved to {output_dir}")
        print(f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")
        return train_df, eval_df, holdout_df
        
    

if __name__=="__main__":
    load_and_Split_data()


        
    