### pipeline/01_data_compiler.py
import pandas as pd
import wbgapi as wb
import time
from utils.config import UNAIDS_RAW, MASTER_PANEL_RAW
from utils.feature_engineering import get_iso3, clean_numeric, enforce_strict_types

def compile_raw_data():
    print("[1/3] Processing core data and confidence intervals from UNAIDS")
    
    df_raw = pd.read_excel(UNAIDS_RAW, sheet_name='HIV2025Estimates_ByArea', header=4)
    df_base = df_raw.iloc[:, [0, 1, 12, 48]].copy()
    df_base.columns = ["Year", "Country_Raw", "Deaths", "New_Infections"]
    
    xls = pd.ExcelFile(UNAIDS_RAW)
    target_sheet = [s for s in xls.sheet_names if 'ByYear' in s and 'Estimates' in s][0]
    df_bounds = pd.read_excel(UNAIDS_RAW, sheet_name=target_sheet, header=None, skiprows=6)
    df_bounds = df_bounds[[0, 1, 49, 50]]
    df_bounds.columns = ['Year', 'ISO_Raw', 'Incidence_Lower', 'Incidence_Upper']
    
    df_base['Year'] = pd.to_numeric(df_base['Year'], errors='coerce')
    df_bounds['Year'] = pd.to_numeric(df_bounds['Year'], errors='coerce')
    df_base['ISO3'] = df_base['Country_Raw'].apply(get_iso3)
    df_bounds['ISO3'] = df_bounds['ISO_Raw'].apply(get_iso3)
    
    for col in ["New_Infections", "Deaths"]:
        df_base[col] = df_base[col].apply(clean_numeric)
    for col in ["Incidence_Lower", "Incidence_Upper"]:
        df_bounds[col] = df_bounds[col].apply(clean_numeric)
        
    df_unaids = pd.merge(df_base.dropna(subset=["ISO3"]), df_bounds.dropna(subset=["ISO3"]), on=["ISO3", "Year"], how="left")

    print("[2/3] Loading macroeconomic & ART data from World Bank API")   
    max_retries = 3
    wb_data = None
    for attempt in range(max_retries):
        try:
            wb_data = wb.data.DataFrame(
                ["SH.HIV.ARTC.ZS", "NY.GDP.PCAP.CD", "SH.XPD.CHEX.GD.ZS", "SP.POP.TOTL"],
                time=range(1990, 2026), numericTimeKeys=True
            ).reset_index()
            print("  + Data loaded successfully from World Bank!")
            break 
        except Exception as e:
            if attempt < max_retries - 1:
                print("  [!] API timeout. Retrying in 10s")
                time.sleep(10) 
            else:
                raise RuntimeError("World Bank API crash.") from e

    wb_melt = wb_data.melt(id_vars=['economy', 'series'], var_name='Year', value_name='Value')
    wb_pivot = wb_melt.pivot_table(index=['economy', 'Year'], columns='series', values='Value').reset_index()
    
    wb_pivot = wb_pivot.rename(columns={
        'economy': 'ISO3', 'SH.HIV.ARTC.ZS': 'ART_Coverage', 'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'SH.XPD.CHEX.GD.ZS': 'Health_Exp', 'SP.POP.TOTL': 'Population'
    })
    wb_pivot['Year'] = wb_pivot['Year'].astype(int)

    print("[3/3] Assembling spatial-temporal RAW panel (NO IMPUTATION)")
    df_master = pd.merge(df_unaids, wb_pivot, on=["ISO3", "Year"], how="left")
    df_master = enforce_strict_types(df_master)
    
    df_master.to_csv(MASTER_PANEL_RAW, index=False)
    print(f" >>> Completed. Saved strict raw data to: {MASTER_PANEL_RAW.name}")

if __name__ == "__main__":
    compile_raw_data()