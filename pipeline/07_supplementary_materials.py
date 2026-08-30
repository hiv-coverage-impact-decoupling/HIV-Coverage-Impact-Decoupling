### pipeline/07_supplementary_materials.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from pathlib import Path
from utils.config import (MASTER_PANEL_TRAIN, FROZEN_EDI_HISTORICAL, MODEL_FORMULA, 
                          MODEL_REQUIRED, PUB_DIR, N_MONTE_CARLO, RANDOM_SEED)
from utils.edi import calculate_pearson_residual
import warnings

warnings.filterwarnings("ignore")

def run_monte_carlo_uncertainty():
    print("\n" + "="*80)
    print(f" [07] MONTE CARLO UNCERTAINTY PROPAGATION ({N_MONTE_CARLO} ITERATIONS) ")
    print("="*80)
    
    # 1. Load Data
    df_train = pd.read_csv(MASTER_PANEL_TRAIN)
    df_frozen_hist = pd.read_csv(FROZEN_EDI_HISTORICAL).set_index('ISO3')['Mean_EDI_2010_2015']
    
    cols = MODEL_REQUIRED + ['Incidence_Lower', 'Incidence_Upper']
    df_mc = df_train.dropna(subset=cols).copy()
    df_mc = df_mc[df_mc['Population'] > 0]
    
    # Estimate standard error of incidence from UNAIDS 95% Confidence Intervals
    df_mc['Incidence_SD'] = (df_mc['Incidence_Upper'] - df_mc['Incidence_Lower']) / 3.92
    
    spearman_scores = []
    failures = 0
    rng = np.random.default_rng(RANDOM_SEED)
    
    print("  [*] Propagating epidemiological uncertainty through the NB2 benchmark")
    
    for i in range(N_MONTE_CARLO):
        # Perturb data: Add Gaussian noise bounded at 0
        sim_inc = rng.normal(loc=df_mc['New_Infections'], scale=df_mc['Incidence_SD'])
        df_mc['Sim_Infections'] = np.maximum(sim_inc, 0)
        
        try:
            # Re-fit Model
            sim_formula = MODEL_FORMULA.replace('New_Infections', 'Sim_Infections')
            mod_sim = smf.negativebinomial(sim_formula, data=df_mc, offset=df_mc['Log_Pop']).fit(disp=False, method='bfgs', maxiter=50)
            
            alpha_sim = mod_sim.params.get('alpha', 0.001)
            pred_sim = mod_sim.predict()
            
            # Recalculate EDI
            df_mc['EDI_Sim'] = calculate_pearson_residual(df_mc['Sim_Infections'], pred_sim, alpha_sim)
            
            # Aggregate to historical window
            sim_hist = df_mc[(df_mc['Year'] >= 2010) & (df_mc['Year'] <= 2015)].groupby('ISO3')['EDI_Sim'].mean()
            
            # Compare with Frozen EDI
            common = df_frozen_hist.index.intersection(sim_hist.index)
            rho, _ = spearmanr(df_frozen_hist[common], sim_hist[common])
            
            if np.isfinite(rho):
                spearman_scores.append(rho)
            else:
                failures += 1
                
        except Exception:
            failures += 1 

    # 2. Compute Stability Metrics
    mean_rho = np.mean(spearman_scores)
    median_rho = np.median(spearman_scores)
    ci_lower = np.percentile(spearman_scores, 2.5)
    ci_upper = np.percentile(spearman_scores, 97.5)
    prop_robust = np.mean(np.array(spearman_scores) > 0.8) * 100
    
    print(f"  + Successful iterations : {len(spearman_scores)}/{N_MONTE_CARLO}")
    print(f"  + Mean Spearman ρ       : {mean_rho:.4f}")
    print(f"  + Median Spearman ρ     : {median_rho:.4f}")
    print(f"  + 95% CI                : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  + Proportion (ρ > 0.8)  : {prop_robust:.1f}%")

    # 3. Export
    out_dir = PUB_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "MonteCarlo.csv"
    
    pd.DataFrame({'Spearman_Rho': spearman_scores}).to_csv(out_csv, index=False)
    print(f"\n>>> Results exported to {out_csv.name}")
    print("="*80)

if __name__ == "__main__":
    run_monte_carlo_uncertainty()