### robustness/mod_econ.py
import pandas as pd
import statsmodels.formula.api as smf
from utils.config import MASTER_PANEL_TRAIN, PUB_DIR

def run_audit():
    df = pd.read_csv(MASTER_PANEL_TRAIN).dropna()
    specs = [
        "New_Infections ~ ART_lag_2 + Year_Index",
        "New_Infections ~ ART_lag_2 + Year_Index + Log_GDP",
        "New_Infections ~ ART_lag_2 + Year_Index + Log_GDP + Health_Exp",
        "New_Infections ~ ART_lag_2 + Year_Index + Log_GDP + Health_Exp + Log_Inf_Rate_lag_1"
    ]
    
    rows = []
    for f in specs:
        try:
            mod = smf.negativebinomial(f, data=df, offset=df["Log_Pop"]).fit(cov_type='HC3', disp=False)
            ci = mod.conf_int().loc["ART_lag_2"]
            rows.append({
                "Specification": f, "Beta_ART": mod.params["ART_lag_2"],
                "CI_Low": ci[0], "CI_High": ci[1], "P_Value": mod.pvalues["ART_lag_2"]
            })
        except Exception as e:
            print(f"      [!] Dropped specification due to non-convergence: {f}")
            
    out_path = PUB_DIR / "robustness" / "SpecificationCurve.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("  [+] Specification Curve with HC3 errors generated safely.")