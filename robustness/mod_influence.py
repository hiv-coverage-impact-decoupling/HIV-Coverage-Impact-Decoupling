### robustness/mod_influence.py
import pandas as pd
import statsmodels.formula.api as smf
from utils.config import PUB_DIR, PRIMARY_VALIDATION_OUTCOME, PRIMARY_VALIDATION_BASELINE, PRIMARY_EDI

def run_audit():
    df = pd.read_csv(PUB_DIR / "main_results" / "Predictive_Validation.csv").reset_index(drop=True)
    formula = f"{PRIMARY_VALIDATION_OUTCOME} ~ {PRIMARY_VALIDATION_BASELINE} + {PRIMARY_EDI}"
    ols_base = smf.ols(formula, data=df).fit()
    
    infl = ols_base.get_influence()
    df['Cooks_D'] = infl.cooks_distance[0]
    
    top_cooks = df.nlargest(3, 'Cooks_D')[['ISO3', 'Cooks_D']]
    print("  [+] Top Influential Observations (Cook's D):")
    for _, row in top_cooks.iterrows():
        print(f"      - {row['ISO3']}: {row['Cooks_D']:.3f}")
    
    out_dir = PUB_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    df[['ISO3', 'Cooks_D']].to_csv(out_dir / "Influence_CooksD.csv", index=False)