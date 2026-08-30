import numpy as np
import pandas as pd
from utils.edi import calculate_pearson_residual

def test_pearson_formula():
    y_true = pd.Series([100, 50, 0])
    y_pred = pd.Series([90, 50, 10])
    alpha = 0.1
    
    res = calculate_pearson_residual(y_true, y_pred, alpha)
    
    # Test observed == predicted -> EDI = 0
    assert np.isclose(res.iloc[1], 0)
    # Test observed > predicted -> positive EDI
    assert res.iloc[0] > 0
    # Test observed < predicted -> negative EDI
    assert res.iloc[2] < 0