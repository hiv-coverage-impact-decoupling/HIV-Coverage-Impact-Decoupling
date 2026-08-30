import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from utils.config import MASTER_PANEL_TRAIN

def run_audit():
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna()
    base = smf.negativebinomial("New_Infections ~ Log_GDP + Health_Exp + ART_lag_2", data=df, offset=df['Log_Pop']).fit(disp=False)
    restr = smf.glm("New_Infections ~ Log_GDP + Health_Exp", data=df, offset=df['Log_Pop'], 
                    family=sm.families.NegativeBinomial(alpha=base.params.get('alpha', 0.05))).fit()
    
    lr_stat = 2 * (base.llf - restr.llf)
    p_val = chi2.sf(lr_stat, base.df_model - restr.df_model)
    print(f"  [+] Likelihood Ratio Test (ART Addition): p = {p_val:.4e}")