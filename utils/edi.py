### utils/edi.py
import numpy as np
import pandas as pd
from utils.config import HIST_START, HIST_END, EDI_WINSOR_LOW, EDI_WINSOR_HIGH

def calculate_pearson_residual(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> pd.Series:
    variance = y_pred + alpha * (y_pred ** 2)
    return (y_true - y_pred) / np.sqrt(variance + 1e-8)

def aggregate_historical_edi(df_yearly: pd.DataFrame) -> pd.DataFrame:
    mask = (df_yearly['Year'] >= HIST_START) & (df_yearly['Year'] <= HIST_END)
    df_hist = df_yearly[mask].groupby("ISO3").agg(
        Mean_EDI_2010_2015=('EDI_Raw', 'mean'),
        SD_EDI_2010_2015=('EDI_Raw', 'std'),
        N_Years=('EDI_Raw', 'count'),
        Mean_ART_2010_2015=('ART_Coverage', 'mean')
    ).reset_index()
    return df_hist

def winsorize_for_display(edi_series: pd.Series) -> pd.Series:
    lower_bound = edi_series.quantile(EDI_WINSOR_LOW)
    upper_bound = edi_series.quantile(EDI_WINSOR_HIGH)
    return edi_series.clip(lower_bound, upper_bound)