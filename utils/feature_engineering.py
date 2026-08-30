### utils/feature_engineering.py
import re
import numpy as np
import pandas as pd
import pycountry
from utils.config import ISO_OVERRIDES, NUMERIC_COLS, ART_LAG, INCIDENCE_LAG

def get_iso3(country_name: str) -> str:
    if pd.isna(country_name): return pd.NA 
    name_clean = str(country_name).strip()
    if name_clean in ISO_OVERRIDES: return ISO_OVERRIDES[name_clean] 
    try:
        result = pycountry.countries.search_fuzzy(name_clean) 
        return result[0].alpha_3 if result else pd.NA 
    except Exception:
        return pd.NA 

def clean_numeric(x, return_na=True): 
    if pd.isna(x): return pd.NA if return_na else 0.0
    s = str(x).replace(" ", "").replace(",", "").replace("<", "").replace(">", "").strip().lower()
    if s in ("", "...", "-"): return pd.NA if return_na else 0.0
    nums = re.findall(r'\d+\.?\d*', s)
    if not nums: return pd.NA if return_na else 0.0
    try:
        val = float(nums[0])
        if s.endswith("m") or "m" in s: val *= 1_000_000
        return val
    except ValueError:
        return pd.NA if return_na else 0.0

def build_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ISO3", "Year"]).copy() 

    df["ART_lag_2"] = df.groupby("ISO3")["ART_Coverage"].shift(2) 
    df["Inf_lag_1"] = df.groupby("ISO3")["New_Infections"].shift(1) 
    df["Pop_lag_1"] = df.groupby("ISO3")["Population"].shift(1) 
    
    df["Year_Index"] = df["Year"] - 1990 
    
    df["Log_Pop"] = np.log(df["Population"].replace(0, np.nan)) 
    df["Log_GDP"] = np.log1p(df["GDP_Per_Capita"].fillna(0))

    inc_rate_lag = (df["Inf_lag_1"] / df["Pop_lag_1"]) * 100_000 
    
    df["Log_Inf_Rate_lag_1"] = np.log1p(inc_rate_lag)
    df["Incidence_Rate"]     = (df["New_Infections"] / df["Population"]) * 100_000 

    return df

def enforce_strict_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") 
    return df