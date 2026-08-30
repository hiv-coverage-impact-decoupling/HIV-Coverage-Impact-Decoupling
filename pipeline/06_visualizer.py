### pipeline/06_visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from utils.config import DECOUPLING_MATRIX, PUB_DIR

FIG_DIR = PUB_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def set_apa_style():
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Open Sans', 'Source Sans Pro', 'Arial'], 
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 12, 
        'axes.titlesize': 14, 
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': False,
        'figure.facecolor': 'white'
    })

def plot_apa_map_distribution():
    print("[*] Rendering Figure 1: Global EDI Distribution (APA Format)")
    
    df_matrix = pd.read_csv(DECOUPLING_MATRIX)
    
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    world = world[(world['POP_EST'] > 0) & (world['ADMIN'] != "Antarctica")]
    
    world = world.merge(df_matrix, how="left", left_on="ISO_A3", right_on="ISO3")

    set_apa_style()
    
    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(top=0.88, bottom=0.15, hspace=0.3) 
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[70, 30])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # === APA FIGURE NUMBER & TITLE ===
    fig.text(0.05, 0.95, "Figure 1", fontsize=12, fontweight='bold', ha='left')
    fig.text(0.05, 0.92, "Global Distribution of the Epidemiological Divergence Index (2010–2015)", 
             fontsize=12, style='italic', ha='left')

    # === [A] Map ===
    world.boundary.plot(ax=ax1, linewidth=0.3, color='black')

    vmin = df_matrix['Mean_EDI_2010_2015'].quantile(0.02)
    vmax = df_matrix['Mean_EDI_2010_2015'].quantile(0.98)
    
    cmap = plt.get_cmap('RdBu_r')
    
    world.plot(column='Mean_EDI_2010_2015', ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax,
               legend=False, missing_kwds={'color': '#E0E0E0'})
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax1, fraction=0.025, pad=0.02)
    cbar.set_label('Mean EDI (2010–2015)', fontsize=11)
    cbar.outline.set_visible(False)
    ax1.axis('off')

    sns.histplot(df_matrix['Mean_EDI_2010_2015'], kde=True, ax=ax2, color='#555555', bins=40, edgecolor='white')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    y_max = ax2.get_ylim()[1]
    ax2.set_ylim(top=y_max * 1.2) 
    ax2.text(0.1, y_max * 1.05, 'Model Alignment (EDI = 0)', color='red', fontsize=10)
    ax2.set_xlabel("Epidemiological Divergence Index (EDI)", fontsize=11)
    ax2.set_ylabel("Number of Countries", fontsize=11)

    # === APA FIGURE NOTE ===
    note_text = ("Note. ")
    fig.text(0.05, 0.05, note_text, fontsize=10, ha='left', wrap=True)

    out_path = FIG_DIR / "Figure1_Global_EDI.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  => Rendered successfully: {out_path.name}")

def plot_apa_decoupling_matrix():
    print("[*] Rendering Figure 2: Coverage-Impact Decoupling Matrix (APA Format)...")
    
    df = pd.read_csv(DECOUPLING_MATRIX)
    set_apa_style()
    
    df['Bubble_Size'] = np.sqrt(df['Mean_New_Infections'].fillna(0)) * 2

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(top=0.88, bottom=0.15) 

    fig.text(0.05, 0.95, "Figure 2", fontsize=12, fontweight='bold', ha='left')
    fig.text(0.05, 0.91, "Coverage–Impact Decoupling Matrix for Global HIV Surveillance", 
             fontsize=12, style='italic', ha='left')

    # === Scatter Plot ===
    med_art = df['Mean_ART_2010_2015'].median()
    med_edi = df['Mean_EDI_2010_2015'].median()

    color_map = {
        'Q1 (Expected Alignment)': '#0072B2',      
        'Q2 (Unexpected Resilience)': '#E69F00',   
        'Q3 (Expected Vulnerability)': '#F0E442',  
        'Q4 (Surveillance Blind Spot)': '#CC79A7'  
    }
    shape_map = {
        'Q1 (Expected Alignment)': 'o',     
        'Q2 (Unexpected Resilience)': 's',  
        'Q3 (Expected Vulnerability)': '^', 
        'Q4 (Surveillance Blind Spot)': 'D' 
    }

    for quad, group_df in df.groupby('Quadrant'):
        ax.scatter(group_df['Mean_ART_2010_2015'], group_df['Mean_EDI_2010_2015'], 
                   s=group_df['Bubble_Size'], 
                   c=color_map[quad], 
                   marker=shape_map[quad],
                   label=quad, 
                   alpha=0.65, 
                   edgecolor='black', 
                   linewidth=0.5)
        
    ax.axvline(med_art, color='#888888', linestyle='--', zorder=0)
    ax.axhline(med_edi, color='#888888', linestyle='--', zorder=0)

    text_props = dict(fontsize=10, weight='bold', alpha=0.7, ha='center', va='center')
    ax.text(med_art + 25, med_edi + 1.5, "Quadrant IV", **text_props, color=color_map['Q4 (Surveillance Blind Spot)'])
    ax.text(med_art - 25, med_edi + 1.5, "Quadrant III", **text_props, color=color_map['Q3 (Expected Vulnerability)'])
    ax.text(med_art - 25, med_edi - 1.5, "Quadrant II", **text_props, color=color_map['Q2 (Unexpected Resilience)'])
    ax.text(med_art + 25, med_edi - 1.5, "Quadrant I", **text_props, color=color_map['Q1 (Expected Alignment)'])

    q4_df = df[df['Quadrant'] == 'Q4 (Surveillance Blind Spot)']
    top_q4 = q4_df.nlargest(5, 'Mean_EDI_2010_2015')
    for _, row in top_q4.iterrows():
        ax.annotate(row['ISO3'], (row['Mean_ART_2010_2015'], row['Mean_EDI_2010_2015']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    ax.set_xlabel("Mean ART Coverage 2010–2015 (%)", fontsize=11)
    ax.set_ylabel("Epidemiological Divergence Index (EDI)", fontsize=11)
    
    # Legend
    leg = ax.legend(title="Surveillance Categories", loc='lower right', fontsize=10) 
    leg.get_title().set_fontsize(10)
    leg.get_title().set_weight('bold')
    for handle in leg.legend_handles:
        handle.set_sizes([80])

    # === APA FIGURE NOTE ===
    note_text = ("Note. ")
    fig.text(0.05, 0.05, note_text, fontsize=10, ha='left', wrap=True)

    out_path = FIG_DIR / "Figure2_Decoupling_Matrix.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  => Rendered successfully: {out_path.name}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" [06] DATA VISUALIZATION (APA 7th Edition) ")
    print("="*80)
    plot_apa_map_distribution()
    plot_apa_decoupling_matrix()
    print("="*80)