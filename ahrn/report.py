from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def tear_sheet(outdir, fold_metrics, monthly, conclusions):
    out=Path(outdir); fm=pd.DataFrame(fold_metrics); mo=pd.DataFrame(monthly)
    fm.to_csv(out/"fold_scoreboard.csv",index=False);mo.to_csv(out/"monthly_returns.csv",index=False)
    fig,ax=plt.subplots(2,2,figsize=(13,8)); sns.barplot(data=fm,x="model",y="net_sharpe",ax=ax[0,0]);ax[0,0].tick_params(axis='x',rotation=70);ax[0,0].set_title("Fold net Sharpe")
    sns.barplot(data=fm,x="model",y="max_drawdown",ax=ax[0,1]);ax[0,1].tick_params(axis='x',rotation=70);ax[0,1].set_title("Maximum drawdown")
    for model,g in mo.groupby("model"): ax[1,0].plot(pd.to_datetime(g.date), (1+g.net_return).cumprod(),label=model)
    ax[1,0].legend(fontsize=6,ncol=2);ax[1,0].set_title("OOS cumulative net wealth")
    txt="\n".join(f"{k}: {v}" for k,v in conclusions.items());ax[1,1].axis("off");ax[1,1].text(.02,.98,txt,va="top",wrap=True)
    fig.tight_layout();fig.savefig(out/"tear_sheet.pdf");fig.savefig(out/"tear_sheet.png",dpi=180);plt.close(fig)
    (out/"conclusions.json").write_text(json.dumps(conclusions,indent=2,default=str))
