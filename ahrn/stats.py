import numpy as np
from scipy import stats

def sharpe(x):
    x=np.asarray(x); return float(np.mean(x)/np.std(x,ddof=1)*np.sqrt(12)) if len(x)>1 and np.std(x,ddof=1)>0 else np.nan
def max_drawdown(x):
    wealth=np.cumprod(1+np.asarray(x)); return float(np.min(wealth/np.maximum.accumulate(wealth)-1))
def stationary_bootstrap_ci(x, reps=2000, block=6, seed=2026):
    """Politis-Romano stationary-block bootstrap CI for the monthly Sharpe."""
    x=np.asarray(x);rng=np.random.default_rng(seed); vals=[];p=1/block;n=len(x)
    for _ in range(reps):
        idx=[];i=rng.integers(n)
        while len(idx)<n:
            idx.append(i); i=rng.integers(n) if rng.random()<p else (i+1)%n
        vals.append(sharpe(x[idx]))
    return tuple(np.quantile(vals,[.05,.95]))
def dm_test(actual, p1, p2):
    d=(np.asarray(actual)-np.asarray(p1))**2-(np.asarray(actual)-np.asarray(p2))**2
    # Newey-West variance with h=1 for non-overlapping monthly forecasts.
    v=np.var(d,ddof=1)/len(d); stat=np.mean(d)/np.sqrt(v) if v>0 else np.nan
    return float(stat), float(2*stats.norm.sf(abs(stat))) if np.isfinite(stat) else np.nan
def bh(pvalues, q=.10):
    p=np.asarray(pvalues,float); order=np.argsort(p); cutoff=-1
    for rank,i in enumerate(order,1):
        if p[i]<=q*rank/len(p): cutoff=rank
    return np.array([False]*len(p)) if cutoff<0 else np.array([p[i]<=p[order[cutoff-1]] for i in range(len(p))])
def paired_summary(a,b):
    d=np.asarray(a)-np.asarray(b); t,p=stats.ttest_1samp(d,0,nan_policy="omit")
    return {"n":int(np.isfinite(d).sum()),"mean_difference":float(np.nanmean(d)),"wins":int(np.nansum(d>0)),"t_stat":float(t),"p_value":float(p)}
