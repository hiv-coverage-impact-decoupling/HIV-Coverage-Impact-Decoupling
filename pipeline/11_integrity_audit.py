### pipeline/11_integrity_audit.py
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from utils.config import (MASTER_PANEL_TRAIN, FROZEN_EDI_YEARLY, PUB_DIR, 
                          TRAIN_START, TRAIN_END, VALIDATION_START)
from utils.edi import calculate_pearson_residual

def run_integrity_audit():
    print("\n" + "="*80)
    print(" [11] FINAL INTEGRITY AUDIT ")
    print("="*80)
    
    audit_report = {"status": "PASS", "checks": {}}
    critical_failure = False

    try:
        # 1. Temporal Integrity (Zero Leakage Check)
        print("  [*] Checking Temporal Boundaries")
        df_train = pd.read_csv(MASTER_PANEL_TRAIN)
        max_train_year = df_train['Year'].max()
        min_train_year = df_train['Year'].min()
        
        check_temporal = (max_train_year <= TRAIN_END) and (min_train_year >= TRAIN_START)
        audit_report["checks"]["temporal_isolation"] = bool(check_temporal)
        
        if not check_temporal:
            print(f"      [!] FAIL: Training data contains year {max_train_year} (Limit: {TRAIN_END})")
            critical_failure = True
        else:
            print(f"      [✓] PASS: Training strict within {TRAIN_START}-{TRAIN_END}")

        # 2. Model Integrity
        print("  [*] Checking Coefficient Validity")
        coef_path = PUB_DIR / "audit" / "NB2_Primary_Coefficients.csv"
        coefs = pd.read_csv(coef_path, index_col=0)["Coefficient"]
        
        has_nan = coefs.isna().any()
        alpha = coefs.get('alpha', -1)
        check_model = (not has_nan) and (alpha > 0)
        audit_report["checks"]["model_validity"] = bool(check_model)
        
        if not check_model:
            print(f"      [!] FAIL: Invalid coefficients or alpha <= 0 (alpha={alpha})")
            critical_failure = True
        else:
            print("      [✓] PASS: Alpha is strictly positive; no NaN parameters")

        # 3. Mathematical Integrity (EDI Formula Verification)
        print("  [*] Checking Mathematical Consistency of EDI")
        df_yearly = pd.read_csv(FROZEN_EDI_YEARLY)
        
        sample = df_yearly.dropna(subset=['New_Infections', 'Predicted_Infections']).sample(10, random_state=42)
        manual_edi = calculate_pearson_residual(sample['New_Infections'], sample['Predicted_Infections'], alpha)
        
        math_match = np.allclose(sample['EDI_Raw'], manual_edi, atol=1e-5)
        audit_report["checks"]["math_consistency"] = bool(math_match)
        
        has_inf = np.isinf(df_yearly['EDI_Raw']).any()
        if not math_match or has_inf:
            print("      [!] FAIL: EDI formula mismatch or Infinity values detected.")
            critical_failure = True
        else:
            print("      [✓] PASS: Pearson residual derived correctly; no Inf values")

        # 4. Export Artifact Verification
        print("  [*] Checking Publication Package (Including Robustness Suite)...")
        required_files = [
            PUB_DIR / "main_results" / "Decoupling_Matrix.csv",
            PUB_DIR / "main_results" / "Predictive_Validation.csv",
            PUB_DIR / "robustness" / "SpecificationCurve.csv", 
            PUB_DIR / "robustness" / "Influence_CooksD.csv",   
            PUB_DIR / "figures" / "Figure1_Global_EDI.pdf",
            PUB_DIR / "figures" / "Figure2_Decoupling_Matrix.pdf"
        ]
        
        missing_files = [f.name for f in required_files if not f.exists()]
        audit_report["checks"]["package_complete"] = len(missing_files) == 0
        
        if missing_files:
            print(f"      [!] FAIL: Missing critical publication artifacts: {missing_files}")
            critical_failure = True
        else:
            print("      [✓] PASS: All core matrices, robustness files, and figures exist")

    except Exception as e:
        print(f"      [!] SYSTEM ERROR during audit: {e}")
        critical_failure = True

    # 5. Final Verdict
    print("\n" + "-"*80)
    audit_file = PUB_DIR / "audit" / "Integrity_Audit.json"
    
    if critical_failure:
        audit_report["status"] = "FAIL"
        with open(audit_file, "w") as f: json.dump(audit_report, f, indent=4)
        print(" [!!!] PIPELINE INTEGRITY COMPROMISED.")
        print(f" [!!!] Report saved to {audit_file}")
        sys.exit(1)
    else:
        with open(audit_file, "w") as f: json.dump(audit_report, f, indent=4)
        print(" [✓] PIPELINE INTEGRITY VERIFIED.")
        print(f" [✓] Saved to {audit_file}")
    print("="*80)

if __name__ == "__main__":
    run_integrity_audit()