# FILE: pipeline/10_supplementary_extras.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.config import PUB_DIR

FIG_DIR = PUB_DIR / "figures"

def set_apa_style():
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.frameon': False
    })

def render_specification_curve():
    csv_path = PUB_DIR / "robustness" / "SpecificationCurve.csv"
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    
    set_apa_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.text(0.05, 0.95, "Figure S1", fontweight='bold', fontsize=12)
    fig.text(0.05, 0.90, "Specification Curve Analysis for ART Effect", style='italic', fontsize=12)
    
    x_pos = np.arange(len(df))
    ax.errorbar(x_pos, df['Beta_ART'], yerr=[df['Beta_ART'] - df['CI_Low'], df['CI_High'] - df['Beta_ART']], 
                fmt='o', color='#1F77B4', capsize=4, markersize=7)
    ax.axhline(0, color='#D62728', linestyle='--', lw=1.5)
    
    labels = ["Model 1\n(Base)", "Model 2\n(+GDP)", "Model 3\n(+Health)", "Model 4\n(+Lag)"]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Regression Coefficient (β)")
    
    fig.text(0.05, 0.05, "Note. Error bars represent 95% confidence intervals with HC3 robust standard errors.", fontsize=10)
    plt.savefig(FIG_DIR / "FigureS1_SpecCurve.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def render_monte_carlo():
    csv_path = PUB_DIR / "robustness" / "MonteCarlo.csv"
    if not csv_path.exists(): return
    df = pd.read_csv(csv_path)
    
    set_apa_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.88, bottom=0.2)
    fig.text(0.05, 0.95, "Figure S2", fontweight='bold', fontsize=12)
    fig.text(0.05, 0.90, "Monte Carlo Distribution of EDI Rank Stability", style='italic', fontsize=12)
    
    sns.histplot(df['Spearman_Rho'], kde=True, color='#7F7F7F', ax=ax, bins=35)
    mean_val = df['Spearman_Rho'].mean()
    ax.axvline(mean_val, color='#D62728', linestyle='--', label=f'Mean = {mean_val:.3f}')
    
    ax.set_xlabel("Spearman Rank Correlation (ρ)")
    ax.set_ylabel("Density")
    ax.legend(loc='upper left')
    
    fig.text(0.05, 0.05, "Note. Distribution of rank correlations across 1,000 simulations with empirical uncertainty propagation.", fontsize=10)
    plt.savefig(FIG_DIR / "FigureS2_MonteCarlo.pdf", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("[*] Rendering Supplementary Figures")
    render_specification_curve()
    render_monte_carlo()