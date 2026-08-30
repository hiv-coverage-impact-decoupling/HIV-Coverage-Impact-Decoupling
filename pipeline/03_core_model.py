### pipeline/03_core_model.py
import pandas as pd
import statsmodels.formula.api as smf
import json
from pathlib import Path
from utils.config import (MASTER_PANEL_TRAIN, MODEL_FORMULA, MODEL_REQUIRED, 
                          FROZEN_EDI_YEARLY, FROZEN_EDI_HISTORICAL, PUB_DIR)
from utils.edi import calculate_pearson_residual, aggregate_historical_edi

def run_core_model():
    print("\n" + "="*80)
    print(" [03] PRIMARY NB2 BENCHMARK & EDI FREEZE (1990-2015) ")
    print("="*80)
    
    # 1. Load Data
    df_train = pd.read_csv(MASTER_PANEL_TRAIN)
    df_model = df_train.dropna(subset=MODEL_REQUIRED).copy()
    df_model = df_model[df_model["Population"] > 0]
    
    print(f"[*] Training strictly on N = {len(df_model)} observations (1990-2015).")
    
    # 2. Fit NB2 Model
    print("[*] Fitting Negative Binomial (NB2) Benchmark")
    model = smf.negativebinomial(MODEL_FORMULA, data=df_model, offset=df_model["Log_Pop"])
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df_model["ISO3"]}, disp=False, method='bfgs')
    
    alpha_est = res.params.get('alpha', 0.001)
    df_model['Predicted_Infections'] = res.predict()
    
    # 3. Calculate EDI (Single Source of Truth)
    df_model['EDI_Raw'] = calculate_pearson_residual(df_model['New_Infections'], df_model['Predicted_Infections'], alpha_est)
    
    # 4. Filter to Historical Window (2010-2015)
    df_yearly = df_model[(df_model['Year'] >= 2010) & (df_model['Year'] <= 2015)].copy()
    
    # 5. Export Frozen Yearly EDI
    cols_to_keep = ['ISO3', 'Country_Raw', 'Year', 'New_Infections', 'Predicted_Infections', 'EDI_Raw', 'ART_Coverage']
    df_yearly = df_yearly[cols_to_keep]
    df_yearly.to_csv(FROZEN_EDI_YEARLY, index=False)
    
    # 6. Aggregate to Country Level & Export Frozen Historical EDI
    df_historical = aggregate_historical_edi(df_yearly)
    
    country_names = df_yearly[['ISO3', 'Country_Raw']].drop_duplicates()
    df_historical = pd.merge(df_historical, country_names, on='ISO3', how='left')
    df_historical.to_csv(FROZEN_EDI_HISTORICAL, index=False)
    
    # 7. Audit Logging
    audit_dir = PUB_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    with open(audit_dir / "NB2_Primary_Model_Summary.txt", "w") as f:
        f.write(res.summary().as_text())
        
    res.params.to_csv(audit_dir / "NB2_Primary_Coefficients.csv", header=["Coefficient"])
    
    print("\n[+] MODEL COEFFICIENTS:")
    print(res.params.round(4).to_string())
    print(f"\n[+] FROZEN ARTIFACTS GENERATED:")
    print(f"  >> Yearly (2010-2015): {FROZEN_EDI_YEARLY.name}")
    print(f"  >> Aggregate Mean    : {FROZEN_EDI_HISTORICAL.name}")
    print("="*80)

if __name__ == "__main__":
    run_core_model()