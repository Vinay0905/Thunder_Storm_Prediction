import pandas as pd 

def apply_guardrails(df:pd.DataFrame)->pd.DataFrame:
    """
    Automatically corrects common unit errors and 'haywire' data.
    """
    df=df.copy()
    # 1. Automatic Unit Correction for Thickness
    # If a user enters 570 (decameters), we convert it to 5700 (meters)
    if 'thickness_1000_500' in df.columns:
        mask = df['thickness_1000_500'] < 1000
        df.loc[mask, 'thickness_1000_500'] *= 10
        
    # 2. CINE sign correction
    # CINE is energy required (convective inhibition). 
    # In most models, it is represented as a negative number or zero.
    if 'cine' in df.columns:
        df['cine'] = df['cine'].apply(lambda x: -abs(x) if x > 0 else x)
        
    return df


def apply_feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    """
    Standardizes feature names and creates engineered features.
    This function ensures consistency between training and serving.
    """
    df=df.copy()

    df=apply_guardrails(df)

    # 1. Internal Logic for Engineered Features
    # We support both API (lowercase) and Pipeline (uppercase) names
    showalter = df.get('showalter_index', df.get('Showalter index'))
    lifted = df.get('lifted_index', df.get('LIFTED index'))
    cape = df.get('cape', df.get('CAPE'))
    cine = df.get('cine', df.get('CINE'))
    
    # Perform calculations if data exists
    if showalter is not None and lifted is not None:
        df['Environmental Stability'] = showalter + lifted
    
    if cape is not None and cine is not None:
        df['Convective Potential'] = cape + cine
    # 2. Rename columns to match what the 'joblib' model expects
    rename_mapping = {
        'sweat_index': 'SWEAT index',
        'k_index': 'K index',
        'totals_totals_index': 'Totals totals index',
        'precipitable_water': 'Moisture Indices',
        'PRECIPITABLE WATER': 'Moisture Indices',
        'thickness_1000_500': 'Temperature Pressure',
        '1000-500 THICKNESS': 'Temperature Pressure',
        'plcl': 'Moisture Temperature Profiles',
        'PLCL': 'Moisture Temperature Profiles'
    }
    
    # Only apply renaming if the source column exists
    existing_renames = {k: v for k, v in rename_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_renames)
    
    return df