### ./exploratory/epidemic_income_audit.py
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from utils.config import MASTER_PANEL_TRAIN, MODEL_FORMULA, PUB_DIR, MODEL_REQUIRED

def run_exploratory_audit():
    print("\n" + "="*80)
    print(" [EXPLORATORY] EPIDEMIC TYPE & INCOME GROUP SENSITIVITY AUDIT ")
    print("="*80)
    
    # 1. Load Time-Safe Training Data
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna(subset=MODEL_REQUIRED)
    df = df[df['Population'] > 0].copy()
    
    # 2. Add Categorical Classifications
    generalized_iso = ['BWA', 'ZAF', 'SWZ', 'LSO', 'NAM', 'ZMB', 'ZWE', 'MOZ']
    df['Epidemic_Type'] = np.where(df['ISO3'].isin(generalized_iso), 'Generalized', 'Concentrated')
    df['Income_Group'] = pd.qcut(df['Log_GDP'], q=4, labels=['Low', 'Lower_Middle', 'Upper_Middle', 'High'])
    
    # 3. Fit Baseline (Frozen Specification)
    print("  [*] Fitting Baseline Model...")
    mod_base = smf.negativebinomial(MODEL_FORMULA, data=df, offset=df['Log_Pop']).fit(disp=False)
    
    # 4. Fit Extended Model
    print("  [*] Fitting Extended Model (+ Epidemic Type + Income Group)...")
    ext_formula = MODEL_FORMULA + " + C(Epidemic_Type) + C(Income_Group)"
    mod_ext = smf.negativebinomial(ext_formula, data=df, offset=df['Log_Pop']).fit(disp=False)
    
    # 5. Likelihood Ratio Test (LRT)
    lr_stat = 2 * (mod_ext.llf - mod_base.llf)
    df_diff = mod_ext.df_model - mod_base.df_model
    lr_p = chi2.sf(lr_stat, df_diff)
    
    print("\n[A] LIKELIHOOD RATIO TEST RESULTS")
    print(f"  + Base Model LLF     : {mod_base.llf:.2f}")
    print(f"  + Extended Model LLF : {mod_ext.llf:.2f}")
    print(f"  + LRT Statistic      : {lr_stat:.2f} (df={df_diff}, p-value={lr_p:.4e})")
    
    # 6. Extract Coefficients of Interest
    print("\n[B] STRUCTURAL COEFFICIENTS (95% CI)")
    conf_int = mod_ext.conf_int()
    for idx in mod_ext.params.index:
        if 'Epidemic' in idx or 'Income' in idx:
            beta = mod_ext.params[idx]
            pval = mod_ext.pvalues[idx]
            ci_low, ci_high = conf_int.loc[idx]
            print(f"  + {idx:<35}: β={beta:>7.4f}, p={pval:>6.4f} | 95% CI [{ci_low:>7.4f}, {ci_high:>7.4f}]")
            
    # 7. Export detailed summary for Appendix/Reviewers
    out_dir = PUB_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "Exploratory_Epidemic_Income_Summary.txt"
    
    with open(out_file, "w") as f:
        f.write("=== EXTENDED MODEL SUMMARY (FOR REVIEWER RESPONSES) ===\n")
        f.write(mod_ext.summary().as_text())
        
    print(f"\n>>> Exploratory Audit completed. Full summary saved to {out_file.name}")
    print("="*80)

if __name__ == "__main__":
    run_exploratory_audit()