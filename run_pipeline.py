### run_pipeline.py
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Core pipeline
CORE_STEPS = [
    ("pipeline/01_data_compiler.py", "Load Raw Data (UNAIDS & World Bank)"),
    ("pipeline/02_time_safe_preprocessor.py", "Time-Safe Imputation & Training Boundary Enforcement"),
    ("pipeline/03_core_model.py", "Fit NB2 Benchmark & Generate Frozen Historical EDI"),
    ("pipeline/04_decoupling_analysis.py", "Coverage-Impact Decoupling Matrix"),
    ("pipeline/05_validation_suite.py", "Strict Predictive Validation & Temporal Stability"),
    ("pipeline/06_visualizer.py", "Render Main Manuscript Figures (Maps & Matrix)"),
    ("pipeline/09_results_exporter.py", "Export Primary Results to Publication Package")
]

# Robustness pipeline
ROBUSTNESS_STEPS = [
    ("pipeline/07_supplementary_materials.py", "Monte Carlo Uncertainty Propagation"),
    ("pipeline/08_advanced_robustness.py", "Orchestrate Advanced Robustness & Sensitivity Audits"),
    ("pipeline/10_supplementary_extras.py", "Render Supplementary Figures (Spec Curve, Monte Carlo, etc.)")
]

AUDIT_STEP = [
    ("pipeline/11_integrity_audit.py", "Final Integrity Audit & Verification")
]

class Logger(object):
    def __init__(self, filename="Statistical_Calculations_Log.txt"):
        self.terminal = sys.stdout
        audit_dir = BASE_DIR / "publication" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.log = open(audit_dir / filename, "w", encoding="utf-8")
        self.log.write("="*80 + "\n")
        self.log.write("HIV EDI PIPELINE ANALYTICAL FREEZE\n")
        self.log.write("="*80 + "\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def print_header(text):
    print(f"\n{'=' * 80}")
    print(f" {text}")
    print(f"{'=' * 80}")

def run_pipeline():
    run_full = "--full" in sys.argv
    
    if run_full:
        steps_to_run = CORE_STEPS + ROBUSTNESS_STEPS + AUDIT_STEP
        mode_text = "FULL PUBLICATION MODE"
    else:
        steps_to_run = CORE_STEPS + AUDIT_STEP
        mode_text = "MAIN ANALYSIS MODE (FAST - SKIPPING ROBUSTNESS)"
    
    sys.stdout = Logger()
    print_header(f"HIV-1 EDI PIPELINE - {mode_text}")
    start_time = time.time()

    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + custom_env.get("PYTHONPATH", "")

    for script_path, description in steps_to_run:
        full_path = BASE_DIR / script_path
        
        print(f"\n[*] Module is running: {script_path}")
        print(f"[*] Task: {description}")
        print("-" * 50)
        
        if not full_path.exists():
            print(f"[!] NETWORK/SYSTEM ERROR: File not found {full_path}")
            print(f"[!] Pipeline halted to maintain structural integrity.")
            sys.exit(1)

        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                check=True, 
                text=True, 
                capture_output=True, 
                env=custom_env
            )
            print(result.stdout)
            print(f" [✓] Completed: {script_path}")
            
        except subprocess.CalledProcessError as e:
            print("\n" + "!" * 80)
            print(f"[!!!] PIPELINE FAILED AT: {script_path}")
            print("[!!!] INTEGRITY BREACH DETECTED. FIX ERROR BEFORE CONTINUING.")
            print(e.stdout)
            print(e.stderr)
            print("!" * 80)
            sys.exit(1)

    elapsed_time = time.time() - start_time
    print_header(f" PIPELINE SUCCESS (TIMING: {elapsed_time:.2f}s)")
    print("1. PRIMARY RESULTS: publication/main_results/")
    print("2. TERMINAL OUTPUT: publication/audit/Statistical_Calculations_Log.txt")
    print("3. INTEGRITY AUDIT: publication/audit/Integrity_Audit.json")
    
    if not run_full:
        print("\n[*] Note: Tier 2 & Tier 3 Robustness and Monte Carlo simulations were skipped.")
        print("[*] To run the complete publication suite, use: python run_pipeline.py --full")

if __name__ == "__main__":
    run_pipeline()