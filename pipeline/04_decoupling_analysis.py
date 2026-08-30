### pipeline/04_decoupling_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.config import FROZEN_EDI_HISTORICAL, FROZEN_EDI_YEARLY, DECOUPLING_MATRIX

def run_decoupling():
    print("\n" + "="*80)
    print(" [04] COVERAGE-IMPACT DECOUPLING ANALYSIS (FROM FROZEN EDI) ")
    print("="*80)

    # 1. Load Frozen Artifacts
    if not FROZEN_EDI_HISTORICAL.exists() or not FROZEN_EDI_YEARLY.exists():
        raise FileNotFoundError("[!] Error: Frozen EDI data not found. Run 03_core_model.py first.")

    df_hist = pd.read_csv(FROZEN_EDI_HISTORICAL)
    df_yearly = pd.read_csv(FROZEN_EDI_YEARLY)

    # 2. Extract Average Burden (New Infections) for Bubble Sizing in Figure 2
    burden_df = df_yearly.groupby('ISO3')['New_Infections'].mean().reset_index()
    burden_df.rename(columns={'New_Infections': 'Mean_New_Infections'}, inplace=True)
    
    matrix_df = pd.merge(df_hist, burden_df, on='ISO3', how='left')

    # 3. Calculate Global Benchmarks (Medians from the 2010-2015 historical window)
    median_art = matrix_df['Mean_ART_2010_2015'].median()
    median_edi = matrix_df['Mean_EDI_2010_2015'].median()

    print(f"[*] Global Benchmarks Computed:")
    print(f"  + Median ART Coverage : {median_art:.2f}%")
    print(f"  + Median EDI Score    : {median_edi:.4f}")

    # 4. Strictly Classify into 4 Epidemiological Surveillance Quadrants
    matrix_df['Quadrant'] = np.where(
        (matrix_df['Mean_ART_2010_2015'] >= median_art) & (matrix_df['Mean_EDI_2010_2015'] > median_edi), 
        'Q4 (Surveillance Blind Spot)',
        np.where((matrix_df['Mean_ART_2010_2015'] < median_art) & (matrix_df['Mean_EDI_2010_2015'] > median_edi), 
                 'Q3 (Expected Vulnerability)',
        np.where((matrix_df['Mean_ART_2010_2015'] < median_art) & (matrix_df['Mean_EDI_2010_2015'] <= median_edi), 
                 'Q2 (Unexpected Resilience)', 
                 'Q1 (Expected Alignment)'))
    )

    # 5. Calculate Rank Discordance
    matrix_df['ART_Rank'] = matrix_df['Mean_ART_2010_2015'].rank(ascending=False).astype(int)
    matrix_df['EDI_Rank'] = matrix_df['Mean_EDI_2010_2015'].rank(ascending=True).astype(int)
    matrix_df['Discordance_Gap'] = matrix_df['EDI_Rank'] - matrix_df['ART_Rank']

    # 6. Save Final Matrix
    DECOUPLING_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(DECOUPLING_MATRIX, index=False)
    
    # 7. Summary
    print("\n[+] Quadrant Distribution:")
    print(matrix_df['Quadrant'].value_counts().to_string())
    print(f"\n=> Decoupling matrix safely frozen and exported to {DECOUPLING_MATRIX.name}")
    print("="*80)

if __name__ == "__main__":
    run_decoupling()