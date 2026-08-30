import pandas as pd
from utils.config import MASTER_PANEL_TRAIN, MODEL_REQUIRED

def test_missing_lags():
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna(subset=MODEL_REQUIRED)
    assert df['Log_Inf_Rate_lag_1'].isna().sum() == 0, "Missing lags should be explicitly dropped, not imputed."