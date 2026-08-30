### utils/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" 
PUB_DIR = BASE_DIR / "publication"

TRAIN_START = 1990
TRAIN_END = 2015

HIST_START = 2010
HIST_END = 2015

VALIDATION_START = 2016
VALIDATION_END = 2022

ART_LAG = 2
INCIDENCE_LAG = 1
EDI_WINSOR_LOW = 0.01
EDI_WINSOR_HIGH = 0.99

N_MONTE_CARLO = 1000
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

NUMERIC_COLS = ["New_Infections", "ART_Coverage", "Population", "GDP_Per_Capita", "Health_Exp"] 
MODEL_REQUIRED = ["New_Infections", "ART_lag_2", "Year_Index", "Log_Inf_Rate_lag_1", "Log_GDP", "Health_Exp", "Log_Pop"] 
MODEL_FORMULA = "New_Infections ~ ART_lag_2 + Year_Index + Log_Inf_Rate_lag_1 + Log_GDP + Health_Exp" 

UNAIDS_RAW            = DATA_DIR / "HIV Estimates 1990-2025.xlsx" 
MASTER_PANEL_RAW      = DATA_DIR / "ISO_Master_Country_Panel_RAW.csv" 
MASTER_PANEL_TRAIN    = DATA_DIR / "ISO_Master_Country_Panel_TRAIN.csv" 


FROZEN_EDI_YEARLY     = DATA_DIR / "EDI_Yearly_Raw_2010_2015.csv"
FROZEN_EDI_HISTORICAL = DATA_DIR / "EDI_Historical_2010_2015.csv"
DECOUPLING_MATRIX     = PUB_DIR / "main_results" / "Decoupling_Matrix.csv"

ISO_OVERRIDES = {
    "Viet Nam": "VNM", "Vietnam":  "VNM", "USA": "USA", "UK": "GBR", "Russia": "RUS",
}

PRIMARY_VALIDATION_OUTCOME = "Delta_Incidence_Rate"
PRIMARY_VALIDATION_BASELINE = "Baseline_Incidence_Rate"
PRIMARY_EDI = "Mean_EDI_2010_2015"