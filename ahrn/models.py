from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from hmmlearn.hmm import GaussianHMM
try:
    from xgboost import XGBRegressor
except ImportError: XGBRegressor = None

def fit_hmm(train: pd.DataFrame, apply: pd.DataFrame, states: int):
    cols = [c for c in ["vix","move","credit_spread","curve_2y10y"] if c in train]
    tr = train.drop_duplicates("month").sort_values("date")[cols].ffill().bfill()
    ev = apply.drop_duplicates("month").sort_values("date")[cols].ffill().bfill()
    hmm = GaussianHMM(n_components=states, covariance_type="diag", n_iter=200, random_state=2026)
    hmm.fit(tr.to_numpy())
    p = pd.DataFrame(hmm.predict_proba(ev.to_numpy()), index=ev.month, columns=[f"regime_{i}" for i in range(states)])
    return p, hmm

class SequenceNet(nn.Module):
    def __init__(self, f, hidden, kind="lstm"):
        super().__init__(); self.rnn = (nn.GRU if kind=="gru" else nn.LSTM)(f, hidden, batch_first=True); self.out=nn.Linear(hidden,1)
    def forward(self,x,p=None): return self.out(self.rnn(x)[0][:,-1]).squeeze(-1)

class AttentionNet(nn.Module):
    """Cross-sectional self-attention followed by regime-gated forecast experts."""
    def __init__(self, f, hidden, heads, states, gated=True, dropout=.1):
        super().__init__(); self.states,self.gated=states,gated
        self.rnn=nn.GRU(f,hidden,batch_first=True); self.attn=nn.MultiheadAttention(hidden,heads,batch_first=True,dropout=dropout)
        self.norm=nn.LayerNorm(hidden); self.experts=nn.Linear(hidden,states if gated else 1)
    def forward(self,x,p):
        h=self.rnn(x)[0][:,-1].unsqueeze(0)       # all stocks in this rebalance date are attention tokens
        a,_=self.attn(h,h,h,need_weights=False); z=self.norm(h+a).squeeze(0); e=self.experts(z)
        return (e*p).sum(1) if self.gated else e.squeeze(1)

def _tensor(x): return torch.as_tensor(x, dtype=torch.float32)
def fit_torch(net, seq, y, probs, costs, month, cfg, variant="A1"):
    torch.manual_seed(cfg["seed"]); opt=torch.optim.AdamW(net.parameters(),lr=cfg["model"]["learning_rate"],weight_decay=cfg["model"]["weight_decay"])
    groups=[np.where(month==m)[0] for m in np.unique(month)]
    for _ in range(cfg["model"]["epochs"]):
        np.random.shuffle(groups); net.train()
        for ix in groups:
            if len(ix)<2: continue
            pred=net(_tensor(seq[ix]),_tensor(probs[ix])); yy=_tensor(y[ix]); loss=nn.functional.mse_loss(pred,yy)
            # Role B: training-only regime-dependent cost regularisation. Role A is untouched.
            if variant in {"A2","A3","A4"}:
                crisis=_tensor(probs[ix][:,-1]); lam=1+(cfg["cost"]["regime_lambda_max"]-1)*crisis
                loss=loss+cfg["cost"]["regularizer"]*torch.mean(lam*_tensor(costs[ix])*torch.abs(pred))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
    return net

def predict_torch(net, seq, probs, month):
    out=np.zeros(len(seq));net.eval()
    with torch.no_grad():
        for m in np.unique(month):
            ix=np.where(month==m)[0];out[ix]=net(_tensor(seq[ix]),_tensor(probs[ix])).cpu().numpy()
    return out

class FittedModel:
    def __init__(self, name, cfg): self.name,self.cfg=name,cfg
    def fit(self, train, X, seq, probs):
        self.features=X.shape[1]; self.mean=float(train.target.mean()); n=self.name
        if n=="B0": return self
        if n=="B2": return self
        if n in {"B1","I1"}: self.model=Ridge(alpha=10.).fit(np.c_[X,probs] if n=="I1" else X,train.target);return self
        if n=="B3":
            self.models=[]
            for k in range(probs.shape[1]):
                w=probs[:,k];self.models.append(Ridge(alpha=10.).fit(X,train.target,sample_weight=w))
            return self
        if n=="B5_RF": self.model=RandomForestRegressor(n_estimators=300,max_features=.7,min_samples_leaf=10,n_jobs=-1,random_state=self.cfg["seed"]).fit(X,train.target);return self
        if n=="B5_XGB":
            if XGBRegressor is None: raise ImportError("Install xgboost for B5_XGB")
            self.model=XGBRegressor(n_estimators=400,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,n_jobs=-1,random_state=self.cfg["seed"]).fit(X,train.target);return self
        if n in {"B4_LSTM","B4_GRU"}:
            self.net=fit_torch(SequenceNet(seq.shape[2],self.cfg["model"]["hidden"],"gru" if n.endswith("GRU") else "lstm"),seq,train.target.to_numpy(),probs,train.cost_role_a.to_numpy(),train.month.to_numpy(),self.cfg);return self
        if n in {"I2","A1","A2","A3","A4"}:
            gated=n!="I2"; self.net=AttentionNet(seq.shape[2],self.cfg["model"]["hidden"],self.cfg["model"]["heads"],probs.shape[1],gated,self.cfg["model"]["dropout"])
            self.net=fit_torch(self.net,seq,train.target.to_numpy(),probs,train.cost_role_a.to_numpy(),train.month.to_numpy(),self.cfg,n);return self
        raise ValueError(n)
    def predict(self, test, X, seq, probs):
        n=self.name
        if n=="B0": return np.repeat(self.mean,len(test))
        if n=="B2": return .35*test.momentum_12_1.fillna(0).to_numpy()-.08*test.price_to_book.fillna(0).to_numpy()+.06*test.oper_margin.fillna(0).to_numpy()
        if n in {"B1","I1","B5_RF","B5_XGB"}: return self.model.predict(np.c_[X,probs] if n=="I1" else X)
        if n=="B3": return sum(probs[:,k]*m.predict(X) for k,m in enumerate(self.models))
        return predict_torch(self.net,seq,probs,test.month.to_numpy())
