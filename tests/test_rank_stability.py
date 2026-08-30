import pandas as pd
from utils.edi import winsorize_for_display

def test_winsorize_preserves_bulk_rank():
    s = pd.Series(range(1, 101))
    w = winsorize_for_display(s)
    assert w.iloc[50] == 51