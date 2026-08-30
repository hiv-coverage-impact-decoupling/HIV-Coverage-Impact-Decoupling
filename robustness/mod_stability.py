### robustness/mod_stability.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from utils.config import MASTER_PANEL_TRAIN, FROZEN_EDI_HISTORICAL, PUB_DIR
from utils.edi import calculate_pearson_residual

def run_audit():
    # 1. Strict Common Denominator
    req_cols = ['New_Infections', 'Population', 'ART_lag_2', 'Year_Index', 'Log_GDP', 'Health_Exp', 'Log_Pop', 'Log_Inf_Rate_lag_1']
    df_train = pd.read_csv(MASTER_PANEL_TRAIN).dropna(subset=req_cols).copy()
    df_train = df_train[df_train['Population'] > 0]
    
    df_frozen = pd.read_csv(FROZEN_EDI_HISTORICAL).set_index('ISO3')['Mean_EDI_2010_2015']
    edi_dict = {'NB_Primary': df_frozen}
    
    # 2. NB2 No Lag
    m_nolag = smf.negativebinomial("New_Infections ~ ART_lag_2 + Year_Index + Log_GDP + Health_Exp", 
                                   data=df_train, offset=df_train['Log_Pop']).fit(disp=False)
    df_train['EDI_NoLag'] = calculate_pearson_residual(df_train['New_Infections'], m_nolag.predict(), m_nolag.params.get('alpha', 0.001))
    edi_dict['NB_NoLag'] = df_train[df_train['Year'] >= 2010].groupby('ISO3')['EDI_NoLag'].mean()
    
    # 3. Poisson (Alpha = 0)
    m_poi = smf.poisson("New_Infections ~ ART_lag_2 + Year_Index + Log_Inf_Rate_lag_1 + Log_GDP + Health_Exp", 
                        data=df_train, offset=df_train['Log_Pop']).fit(disp=False)
    df_train['EDI_Poisson'] = calculate_pearson_residual(df_train['New_Infections'], m_poi.predict(), 0.0)
    edi_dict['Poisson'] = df_train[df_train['Year'] >= 2010].groupby('ISO3')['EDI_Poisson'].mean()
    
    # 4. Rank
    df_rank = pd.DataFrame(edi_dict).dropna()
    print("  [+] Rank Stability vs Primary Benchmark:")
    print(f"      - No Lag Model : Spearman ρ = {spearmanr(df_rank['NB_Primary'], df_rank['NB_NoLag'])[0]:.3f}")
    print(f"      - Poisson Model: Spearman ρ = {spearmanr(df_rank['NB_Primary'], df_rank['Poisson'])[0]:.3f}")
    
    out_dir = PUB_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_rank.corr(method='spearman').to_csv(out_dir / "TemporalStability_Spearman.csv")

if __name__ == "__main__":
    run_audit()