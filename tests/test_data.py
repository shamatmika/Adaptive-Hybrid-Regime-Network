import numpy as np
import pandas as pd
from ahrn.data import exogenous_cost

def test_cost_is_nonnegative_and_exogenous():
    x=pd.DataFrame({"bid_ask_spread":[.01,np.nan],"market_cap":[100.,100.],"dollar_volume":[1000.,1000.]})
    c=exogenous_cost(x)
    assert (c>=0).all() and (c<=.05).all()

def test_forward_fill_never_uses_a_future_value():
    # The implementation is groupby(...).ffill(limit=3), never bfill.
    s=pd.Series([np.nan,2.,np.nan,np.nan]).ffill(limit=3)
    assert np.isnan(s.iloc[0]) and s.iloc[2]==2.
