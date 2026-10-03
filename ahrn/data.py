from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REQUIRED = {"date","ticker","monthly_return","risk_free_rate","in_sp500","volume_avg_30d","bid_ask_spread","market_cap"}

def load_panel(path: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    miss = REQUIRED.union(features).difference(df.columns)
    if miss: raise ValueError(f"Missing required fields: {sorted(miss)}")
    df = df.sort_values(["ticker","date"]).copy()
    # Only use previously published values. A three-month cap reflects quarterly data.
    qcols = [c for c in ["market_cap","price_to_book","px_to_sales","oper_margin","momentum_12_1"] if c in df]
    df[qcols] = df.groupby("ticker", group_keys=False)[qcols].ffill(limit=3)
    df["rf_monthly"] = df["risk_free_rate"].astype(float) / 1200.0
    df["target"] = df.groupby("ticker")["monthly_return"].shift(-1) - df.groupby("ticker")["rf_monthly"].shift(-1)
    # A label may only exist if its next observation is exactly next calendar month.
    nxt = df.groupby("ticker")["date"].shift(-1)
    df.loc[(nxt.dt.to_period("M") - df.date.dt.to_period("M")).apply(lambda x: x.n if pd.notna(x) else 99) != 1, "target"] = np.nan
    df["month"] = df.date.dt.to_period("M").astype(str)
    df["dollar_volume"] = df.get("dollar_volume", np.nan)
    df["cost_role_a"] = exogenous_cost(df)
    return df

def exogenous_cost(df: pd.DataFrame) -> pd.Series:
    half = 0.5 * df.bid_ask_spread.fillna(0.005)
    # A conservative, fully market-data-only proxy. It is identical across models.
    ratio = (df.market_cap / df.dollar_volume.replace(0, np.nan)).clip(0, 1).fillna(0.01)
    impact = 0.0001 * 0.10 * np.sqrt(ratio)
    return (half + impact).clip(0, 0.05)

class FoldPreprocessor:
    def __init__(self, features: list[str]): self.features, self.scaler, self.medians = features, StandardScaler(), None
    def fit(self, train: pd.DataFrame):
        self.medians = train[self.features].median()
        self.scaler.fit(train[self.features].fillna(self.medians))
        return self
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.features] = self.scaler.transform(out[self.features].fillna(self.medians))
        return out

def folds(df: pd.DataFrame, cfg: dict):
    months = np.array(sorted(df.month.unique()))
    wf = cfg["walk_forward"]
    for start in range(0, len(months)-wf["train_months"]-wf["purge_months"], wf["step_months"]):
        train_m = months[start:start+wf["train_months"]]
        test_start = start + wf["train_months"] + wf["purge_months"]
        test_m = months[test_start:test_start+wf["test_months"]]
        if len(test_m) < wf["min_test_months"]: continue
        raw_train = df[df.month.isin(train_m) & (df.in_sp500 == 1)]
        # Pre-registered rule: a name with >20% missing monthly returns is
        # excluded from this complete fold (train and its later test window).
        missing = raw_train.groupby("ticker").monthly_return.apply(lambda x: x.isna().mean())
        eligible = missing[missing <= 0.20].index
        train = raw_train[raw_train.ticker.isin(eligible)].dropna(subset=["target"])
        test = df[df.month.isin(test_m) & (df.in_sp500 == 1) & df.ticker.isin(eligible)].dropna(subset=["target"])
        if train.empty or test.empty: continue
        yield train_m, test_m, train, test

def make_sequences(frame: pd.DataFrame, all_scaled: pd.DataFrame, features: list[str], lookback: int):
    """Return history ending at each row, using data at or before that row only."""
    histories = {t:g.sort_values("date") for t,g in all_scaled.groupby("ticker")}
    out = []
    for r in frame.itertuples():
        h = histories[r.ticker]; h = h[h.date <= r.date].tail(lookback)[features].to_numpy(np.float32)
        if len(h) < lookback: h = np.pad(h, ((lookback-len(h),0),(0,0)))
        out.append(h)
    return np.stack(out)
