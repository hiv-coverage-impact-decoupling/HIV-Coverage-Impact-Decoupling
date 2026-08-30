import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from utils.config import MASTER_PANEL_TRAIN, FROZEN_EDI_HISTORICAL, MODEL_FORMULA

def run_audit():
    southern_africa = ["BWA", "ZAF", "SWZ", "LSO", "NAM", "ZMB", "ZWE", "MOZ"]
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna()
    df_frozen = pd.read_csv(FROZEN_EDI_HISTORICAL)
    
    df_excl = df[~df['ISO3'].isin(southern_africa)].copy()
    mod = smf.negativebinomial(MODEL_FORMULA, data=df_excl, offset=df_excl["Log_Pop"]).fit(disp=False)
    
    from utils.edi import calculate_pearson_residual
    df_excl['EDI_Excl'] = calculate_pearson_residual(df_excl['New_Infections'], mod.predict(), mod.params.get('alpha', 0.001))
    hist_excl = df_excl[df_excl['Year'] >= 2010].groupby('ISO3')['EDI_Excl'].mean().reset_index()
    
    merged = pd.merge(df_frozen, hist_excl, on='ISO3')
    rho, _ = spearmanr(merged['Mean_EDI_2010_2015'], merged['EDI_Excl'])
    print(f"  [+] Regional Exclusion (No Southern Africa) Rank Stability: ρ = {rho:.3f}")