import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from utils.config import MASTER_PANEL_TRAIN

def run_audit():
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna()
    rng = np.random.default_rng(42)
    
    w_grid = [0.0, 0.5, 1.0]
    countries = df["ISO3"].unique()
    
    print("  [+] Simulation-based Circularity Stress Test (Noise=0.15, Effect=-0.1):")
    for w in w_grid:
        rho_list = []
        for _ in range(50): 
            boot_ids = rng.choice(countries, size=len(countries), replace=True)
            boot_df = pd.concat([df[df['ISO3'] == c] for c in boot_ids]).reset_index(drop=True)
            
            true_div = boot_df['ISO3'].map({c: rng.normal(0, 0.5) for c in countries})
            log_true_inc = -4.0 - 0.1*boot_df['Log_GDP'] - 0.05*boot_df['Health_Exp'] + true_div
            log_art = -4.0 - 0.1 * boot_df['ART_lag_2']
            
            log_obs = (1 - w) * log_true_inc + w * log_art + rng.normal(0, 0.15, size=len(boot_df))
            boot_df['Sim_Inc'] = np.exp(log_obs + np.log(boot_df['Population']))
            
            try:
                mod = smf.negativebinomial("Sim_Inc ~ ART_lag_2 + Log_GDP + Health_Exp", data=boot_df, offset=boot_df['Log_Pop']).fit(disp=False, maxiter=50)
                pred = mod.predict()
                edi = (boot_df['Sim_Inc'] - pred) / np.sqrt(pred + mod.params.get('alpha', 0.001)*(pred**2) + 1e-8)
                
                agg_df = pd.DataFrame({'True_Div': true_div, 'EDI': edi, 'ISO3': boot_df['ISO3']}).groupby('ISO3').mean()
                rho, _ = spearmanr(agg_df["True_Div"], agg_df["EDI"], nan_policy="omit")
                if np.isfinite(rho): rho_list.append(rho)
            except: pass
            
        print(f"      - w = {w:.1f} | Median Spearman ρ = {np.median(rho_list):.3f}")