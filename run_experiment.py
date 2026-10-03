from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.linear_model import RidgeCV
from ahrn.data import load_panel, FoldPreprocessor, folds, make_sequences
from ahrn.models import fit_hmm, FittedModel
from ahrn.stats import sharpe, max_drawdown, stationary_bootstrap_ci, dm_test, bh, paired_summary
from ahrn.report import tear_sheet

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True, warn_only=True)

def regime_arrays(train, test, states):
    both=pd.concat([train,test]); p,_=fit_hmm(train,both,states)
    cols=list(p.columns); return train.month.map(p[cols].to_dict("index")).apply(pd.Series).to_numpy(float), test.month.map(p[cols].to_dict("index")).apply(pd.Series).to_numpy(float)

def portfolio(frame, prediction, probs, cfg, model, cost_tier="dynamic"):
    f=frame.copy();f["prediction"]=prediction;f["entropy"]=-np.sum(probs*np.log(np.clip(probs,1e-12,1)),axis=1)
    old={}; output=[]; quint=cfg["portfolio"]["quintiles"]
    for date,g in f.groupby("date",sort=True):
        # The liquidity cut is date-specific and identical for every model.
        g=g[g.volume_avg_30d>=g.volume_avg_30d.quantile(cfg["portfolio"]["illiquid_percentile"])].copy()
        n=len(g);q=n//quint
        if q<2: continue
        g=g.sort_values("prediction");lo,hi=g.iloc[:q],g.iloc[-q:]
        # With a normal S&P-500 quintile there are at least 50 names per leg,
        # so both 100% leg exposure and the pre-declared 2% cap can hold.
        cap=cfg["portfolio"]["position_cap"]; long_w=min(1/q,cap); short_w=-min(1/q,cap)
        w={**{t:long_w for t in hi.ticker}, **{t:short_w for t in lo.ticker}}
        # Role C: only A3/A4 reduce gross size when regime state is uncertain.
        scale=1.0
        if model in {"A3","A4"}: scale=float(1-np.mean(pd.concat([hi.entropy,lo.entropy]))/np.log(probs.shape[1]))
        scale=float(np.clip(scale,0.10,1.0)); w={k:v*scale for k,v in w.items()}
        gross=sum(w[t]*g.set_index("ticker").loc[t,"target"] for t in w)
        turnover=sum(abs(w.get(t,0)-old.get(t,0)) for t in set(w)|set(old))
        costs_df=g.set_index("ticker")
        if cost_tier=="flat":
            base=float(frame.cost_role_a.median()); costs=sum(abs(v)*base for v in w.values())*(turnover/2)
        elif cost_tier=="regime_scaled":
            crisis=float(np.mean(np.r_[hi.entropy,lo.entropy])/np.log(probs.shape[1]))
            costs=sum(abs(v)*costs_df.loc[t,"cost_role_a"] for t,v in w.items())*(turnover/2)*(1+(cfg["cost"]["regime_lambda_max"]-1)*crisis)
        else: costs=sum(abs(v)*costs_df.loc[t,"cost_role_a"] for t,v in w.items())*(turnover/2)
        net=gross-costs-cfg["portfolio"]["short_fee_annual"]/12*scale
        pred_ls=hi.prediction.mean()-lo.prediction.mean()
        output.append(dict(date=date,model=model,cost_tier=cost_tier,gross_return=gross,net_return=net,predicted_ls=pred_ls,turnover=turnover,scale=scale))
        old=w
    return pd.DataFrame(output)

def metrics(monthly, model, fold, start, end):
    r=monthly.net_return.to_numpy();actual=monthly.gross_return.to_numpy();pred=monthly.predicted_ls.to_numpy();base=actual.mean()
    return dict(model=model,fold=fold,test_start=str(start),test_end=str(end),months=len(monthly),net_sharpe=sharpe(r),max_drawdown=max_drawdown(r),oos_r2=float(1-np.sum((actual-pred)**2)/np.sum((actual-base)**2)) if np.sum((actual-base)**2)>0 else np.nan,mean_net_return=float(np.mean(r)),turnover=float(monthly.turnover.mean()),ci90_low=stationary_bootstrap_ci(r)[0],ci90_high=stationary_bootstrap_ci(r)[1])

def save_shap(model, train_x, test_x, feature_names, out, label):
    """Tree SHAP for B5; neural SHAP for A1 is attempted by GradientExplainer."""
    try:
        import shap
        if hasattr(model,"model") and model.name in {"B5_RF","B5_XGB"}:
            values=shap.TreeExplainer(model.model).shap_values(test_x[:500]); pd.DataFrame({"feature":feature_names,"mean_abs_shap":np.abs(values).mean(0)}).sort_values("mean_abs_shap",ascending=False).to_csv(out/f"shap_{label}.csv",index=False)
    except Exception as e: (out/f"shap_{label}_warning.txt").write_text(str(e))

def main(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());seed_everything(cfg["seed"]);out=Path(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=True);shutil.copy(config_path,out/"frozen_config.yaml")
    panel=load_panel(cfg["data"],cfg["features"]); all_metrics=[];all_monthly=[]; fold_predictions={}; manifest={"config":cfg,"rows":len(panel),"models":cfg["models"],"folds":[]}
    for fold,(train_m,test_m,train,test) in enumerate(folds(panel,cfg)):
        prep=FoldPreprocessor(cfg["features"]).fit(train); scaled=prep.transform(panel); tr=scaled.loc[train.index].copy();te=scaled.loc[test.index].copy()
        p_tr,p_te=regime_arrays(tr,te,cfg["model"]["states"]); seq_tr=make_sequences(tr,scaled,cfg["features"],cfg["model"]["lookback"]);seq_te=make_sequences(te,scaled,cfg["features"],cfg["model"]["lookback"])
        Xtr,Xte=tr[cfg["features"]].to_numpy(),te[cfg["features"]].to_numpy();preds_train={};preds_test={};models={}
        # A4 is fitted after its constituent models and contains no OOS data in its meta fit.
        ordinary=[m for m in cfg["models"] if m!="A4"]
        for name in ordinary:
            fitted=FittedModel(name,cfg).fit(tr,Xtr,seq_tr,p_tr);models[name]=fitted
            preds_train[name]=fitted.predict(tr,Xtr,seq_tr,p_tr);preds_test[name]=fitted.predict(te,Xte,seq_te,p_te)
            for tier in ["dynamic","flat","regime_scaled"]:
                mon=portfolio(te,preds_test[name],p_te,cfg,name,tier)
                if len(mon)<cfg["walk_forward"]["min_test_months"]: continue
                all_monthly.extend(mon.assign(fold=fold).to_dict("records"));z=metrics(mon,name,fold,test_m[0],test_m[-1]);z["cost_tier"]=tier;all_metrics.append(z);(out/f"fold_{fold:02d}_{name}_{tier}.json").write_text(json.dumps(z,indent=2))
            if name in {"B5_RF","B5_XGB"}: save_shap(fitted,Xtr,Xte,cfg["features"],out,f"fold{fold}_{name}")
        if "A4" in cfg["models"]:
            bases=[x for x in ["B1","B5_RF","A3"] if x in preds_train]
            # Meta weights are estimated exclusively inside the training fold.
            meta=RidgeCV(alphas=[.1,1.,10.]).fit(np.column_stack([preds_train[x] for x in bases]),tr.target)
            preds_test["A4"]=meta.predict(np.column_stack([preds_test[x] for x in bases]));
            for tier in ["dynamic","flat","regime_scaled"]:
                mon=portfolio(te,preds_test["A4"],p_te,cfg,"A4",tier);all_monthly.extend(mon.assign(fold=fold).to_dict("records"));z=metrics(mon,"A4",fold,test_m[0],test_m[-1]);z["cost_tier"]=tier;all_metrics.append(z);(out/f"fold_{fold:02d}_A4_{tier}.json").write_text(json.dumps({**z,"stack_base_models":bases},indent=2))
        # Gradient SHAP for A1: attribution is reported for every sequence feature.
        try:
            import shap
            if "A1" in models:
                f,k=seq_tr.shape[2],p_tr.shape[1]
                class FlatA1(torch.nn.Module):
                    def __init__(self, net): super().__init__();self.net=net
                    def forward(self,z): return self.net(z[:,:-k].reshape(-1,cfg["model"]["lookback"],f),z[:,-k:])
                wrapper=FlatA1(models["A1"].net); bg=np.c_[seq_tr[:min(80,len(seq_tr))].reshape(min(80,len(seq_tr)),-1),p_tr[:min(80,len(p_tr))]]; ev=np.c_[seq_te[:min(150,len(seq_te))].reshape(min(150,len(seq_te)),-1),p_te[:min(150,len(p_te))]]
                values=shap.GradientExplainer(wrapper,torch.tensor(bg,dtype=torch.float32)).shap_values(torch.tensor(ev,dtype=torch.float32))
                values=np.asarray(values)[:,:-k].reshape(len(ev),cfg["model"]["lookback"],f)
                rows=[]
                for regime in range(k):
                    take=np.argmax(p_te[:len(ev)],axis=1)==regime
                    if take.any():
                        rows.extend({"regime":regime,"feature":name,"mean_abs_shap":float(np.abs(values[take,:,j]).mean())} for j,name in enumerate(cfg["features"]))
                pd.DataFrame(rows).to_csv(out/f"shap_fold{fold}_A1_by_regime.csv",index=False)
        except Exception as e: (out/f"shap_fold{fold}_A1_warning.txt").write_text(str(e))
        fold_predictions[fold]=preds_test;manifest["folds"].append({"fold":fold,"train_start":str(train_m[0]),"train_end":str(train_m[-1]),"test_start":str(test_m[0]),"test_end":str(test_m[-1]),"train_rows":len(tr),"test_rows":len(te)})
    fm=pd.DataFrame(all_metrics);mo=pd.DataFrame(all_monthly)
    fm.to_csv(out/"cost_tier_scoreboard.csv",index=False)
    # Primary inference uses only Role-A dynamic costs. Other tiers are labelled exploratory.
    fm=fm[fm.cost_tier=="dynamic"].copy(); mo_primary=mo[mo.cost_tier=="dynamic"].copy()
    def compare(left,right):
        z=fm[fm.model==left][["fold","net_sharpe","test_start","test_end"]].merge(fm[fm.model==right][["fold","net_sharpe"]],on="fold",suffixes=("_left","_right"))
        return z,paired_summary(z.net_sharpe_left,z.net_sharpe_right)
    a_b1,paired=compare("A1","B1")
    # DM compares realised portfolio returns with each model's predicted long-short spread where aligned.
    dm={}
    for f in sorted(set(mo_primary.fold)):
        x=mo_primary[mo_primary.fold==f];aa=x[x.model=="A1"].sort_values("date");bb=x[x.model=="B1"].sort_values("date")
        if len(aa)==len(bb) and len(aa)>2: dm[str(f)]=dm_test(aa.gross_return,aa.predicted_ls,bb.predicted_ls)
    secondary=["B0","B2","B3","B4_LSTM","B4_GRU","B5_RF","B5_XGB"]; secondary_tests={}
    for name in secondary:
        if name in set(fm.model): secondary_tests[name]=compare("A1",name)[1]
    ablations={}
    for name in ["I1","I2","A2","A3","A4"]:
        if name in set(fm.model):
            z,s=compare("A1",name); s["bonferroni_p_value"]=min(1.,5*s["p_value"]);ablations[f"A1_vs_{name}"]=s
    crisis=[]
    for window in cfg.get("crisis_windows",[]):
        start,end=map(pd.Timestamp,window); crisis.extend(a_b1[(pd.to_datetime(a_b1.test_end)>=start)&(pd.to_datetime(a_b1.test_start)<=end)].fold.tolist())
    crisis=sorted(set(crisis)); leave_one={}
    for f in crisis:
        z=a_b1[a_b1.fold!=f]; s=paired_summary(z.net_sharpe_left,z.net_sharpe_right);leave_one[str(f)]=s
    robust=bool(leave_one) and all(x["mean_difference"]>0 for x in leave_one.values())
    conclusions={"primary_A1_vs_B1":paired,"primary_success":bool(paired["mean_difference"]>0 and paired["wins"]>paired["n"]/2 and paired["p_value"]<.10),"crisis_folds":crisis,"crisis_removal_robust":robust,"leave_one_crisis_fold":leave_one,"diebolt_mariano_by_fold":dm,"secondary_A1_tests":secondary_tests,"BH_q10_rejections":dict(zip(secondary_tests,bh([x["p_value"] for x in secondary_tests.values()]).tolist())),"ablations_bonferroni":ablations,"warning":"A4 meta layer is fold-trained; replace its in-fold base predictions with nested-CV OOF predictions for the final preregistered run."}
    manifest["conclusions"]=conclusions;(out/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str));tear_sheet(out,fm.to_dict("records"),mo_primary.to_dict("records"),conclusions)
    print(json.dumps(conclusions,indent=2,default=str))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);main(p.parse_args().config)
