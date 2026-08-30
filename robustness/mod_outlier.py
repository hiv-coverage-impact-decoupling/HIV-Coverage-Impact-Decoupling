### robustness/mod_outlier.py
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from utils.config import PUB_DIR, PRIMARY_VALIDATION_OUTCOME, PRIMARY_VALIDATION_BASELINE, PRIMARY_EDI

def run_audit():
    df = pd.read_csv(PUB_DIR / "main_results" / "Predictive_Validation.csv")
    formula = f"{PRIMARY_VALIDATION_OUTCOME} ~ {PRIMARY_VALIDATION_BASELINE} + {PRIMARY_EDI}"
    
    # Trimmed (5%)
    lb, ub = df[PRIMARY_EDI].quantile(0.025), df[PRIMARY_EDI].quantile(0.975)
    df_trim = df[(df[PRIMARY_EDI] >= lb) & (df[PRIMARY_EDI] <= ub)]
    res_trim = smf.ols(formula, data=df_trim).fit(cov_type='HC3')
    
    # Huber Robust
    X = sm.add_constant(df[[PRIMARY_EDI, PRIMARY_VALIDATION_BASELINE]])
    res_huber = sm.RLM(df[PRIMARY_VALIDATION_OUTCOME], X, M=sm.robust.norms.HuberT()).fit()
    
    print(f"  [+] Trimmed OLS (HC3) β : {res_trim.params[PRIMARY_EDI]:.4f} (p={res_trim.pvalues[PRIMARY_EDI]:.4f})")
    print(f"  [+] Huber Robust β      : {res_huber.params[PRIMARY_EDI]:.4f} (p={res_huber.pvalues[PRIMARY_EDI]:.4f})")