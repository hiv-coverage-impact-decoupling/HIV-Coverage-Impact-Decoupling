### pipeline/08_advanced_robustness.py
import sys
import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROBUSTNESS_DIR = BASE_DIR / "robustness"

def load_module(name, filename):
    path = ROBUSTNESS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_all_robustness():
    print("\n" + "="*80)
    print(" [08] ADVANCED ROBUSTNESS & SENSITIVITY ORCHESTRATOR ")
    print("="*80)
    
    modules = {
        "mod_stability": "Distributional & Specification Sensitivity (Rank Stability)",
        "mod_econ": "Advanced Econometrics (Spec Curve HC3, First-Diff)",
        "mod_outlier": "Outlier Robustness (Trimmed, Huber)",
        "mod_influence": "Influence Diagnostics (Cook's D, LOCO)",
        "mod_regional": "Regional Rank Robustness (Excl. Southern Africa)",
        "mod_art_lrt": "Nested Model Likelihood Ratio Test",
        "mod_circular": "Simulation-based Circularity Stress Test"
    }
    
    for mod_name, description in modules.items():
        print(f"\n[*] Running: {description}")
        try:
            mod = load_module(mod_name, f"{mod_name}.py")
            if hasattr(mod, 'run_audit'):
                mod.run_audit()
        except Exception as e:
            print(f"\n [!!!] CRITICAL FAILURE IN {mod_name}")
            print(f" [!!!] Error: {e}")
            print(" [!!!] PIPELINE HALTED TO PREVENT FALSE INTEGRITY PASS.")
            sys.exit(1)
            
    print("\n" + "="*80)

if __name__ == "__main__":
    run_all_robustness()