### pipeline/02_time_safe_preprocessor.py
import pandas as pd
from utils.config import MASTER_PANEL_RAW, MASTER_PANEL_TRAIN
from utils.time_safe_preprocessing import time_safe_impute
from utils.feature_engineering import build_model_variables

def preprocess_training_data():
    print("[*] Enforcing Time-Safe Data Preprocessing")
    
    # 1. Load RAW
    df_raw = pd.read_csv(MASTER_PANEL_RAW)
    
    # 2. Strict imputation inside 1990-2015 window
    print("  + Imputing missing values STRICTLY within the 1990-2015 window")
    df_train_imputed = time_safe_impute(df_raw)
    
    # 3. Build lags (calculates lags, takes log, leaves NA as NA)
    print("  + Engineering lagged features (strict NA retention)")
    df_train_engineered = build_model_variables(df_train_imputed)
    
    # 4. Save the cleanly bounded training panel
    df_train_engineered.to_csv(MASTER_PANEL_TRAIN, index=False)
    
    print(f" >>> Complete. Time-safe training panel exported to {MASTER_PANEL_TRAIN.name}")
    print(f" >>> Shape of Training Panel: {df_train_engineered.shape}")

if __name__ == "__main__":
    preprocess_training_data()