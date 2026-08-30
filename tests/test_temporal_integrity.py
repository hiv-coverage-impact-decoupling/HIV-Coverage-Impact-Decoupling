import pandas as pd
from utils.config import MASTER_PANEL_TRAIN, TRAIN_END

def test_training_boundary():
    df = pd.read_csv(MASTER_PANEL_TRAIN)
    assert df['Year'].max() <= TRAIN_END, f"Leakage detected: Found data > {TRAIN_END}"