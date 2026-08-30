### utils/time_safe_preprocessing.py
import pandas as pd
from utils.config import TRAIN_START, TRAIN_END

def time_safe_impute(df_raw: pd.DataFrame) -> pd.DataFrame:
    
    # 1. Isolate temporal window
    df_train = df_raw[(df_raw['Year'] >= TRAIN_START) & (df_raw['Year'] <= TRAIN_END)].copy()
    
    # 2. Impute only using available historical data
    cols_to_impute = ["ART_Coverage", "GDP_Per_Capita", "Health_Exp", "Population"]
    for col in cols_to_impute:
        df_train[col] = df_train.groupby("ISO3")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both").bfill().ffill()
        )
    return df_train