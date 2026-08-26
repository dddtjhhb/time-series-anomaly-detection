"""Run the complete real-data and synthetic-data experiments."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(Path(".mplconfig").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data import combine_assets, load_or_download
from src.evaluation import evaluate_grid
from src.methods import add_rolling_statistics, flag_anomalies
from src.synthetic import make_synthetic_series

def real_experiment(c,refresh):
    frames=[load_or_download(t,c["start_date"],c["end_date"],ROOT/"data/raw",refresh) for t in c["tickers"]]
    prices=combine_assets(frames); prices.to_csv(ROOT/"data/processed/real_returns.csv",index=False)
    rows=[]; example=None
    for ticker,group in prices.groupby("Ticker"):
        for window in c["windows"]:
            scored=add_rolling_statistics(group.reset_index(drop=True),window)
            for threshold in c["thresholds"]:
                flagged=flag_anomalies(scored,threshold)
                valid=int(flagged["ZScore"].notna().sum()); count=int(flagged["IsAnomaly"].sum())
                rows.append({"Ticker":ticker,"Window":window,"Threshold":threshold,"ValidDays":valid,
                             "Anomalies":count,"AnomalyRate":count/valid})
                if (ticker,window,threshold)==("SPY",20,2.5): example=flagged
    pd.DataFrame(rows).to_csv(ROOT/"results/tables/real_grid.csv",index=False)
    fig,ax=plt.subplots(figsize=(11,4.5)); chosen=example["IsAnomaly"]
    ax.plot(example["Date"],example["Return"],color="#476A8A",lw=.7,label="Daily return")
    ax.scatter(example.loc[chosen,"Date"],example.loc[chosen,"Return"],color="#D1495B",s=20,label="|z| > 2.5")
    ax.axhline(0,color="black",lw=.6); ax.set(title="SPY anomalies: 20-day rolling z-score",ylabel="Return")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(ROOT/"results/figures/spy_anomalies.png",dpi=180); plt.close(fig)

def synthetic_experiment(c):
    data=make_synthetic_series(c["synthetic_length"],c["synthetic_anomalies"],c["random_seed"])
    data.to_csv(ROOT/"data/processed/synthetic_returns.csv",index=False)
    grid=evaluate_grid(data,c["windows"],c["thresholds"]); grid.to_csv(ROOT/"results/tables/synthetic_grid.csv",index=False)
    scored=flag_anomalies(add_rolling_statistics(data,60),2.5)
    scored["Case"]="true_negative"
    scored.loc[scored["IsInjected"] & scored["IsAnomaly"],"Case"]="true_positive"
    scored.loc[~scored["IsInjected"] & scored["IsAnomaly"],"Case"]="false_positive"
    scored.loc[scored["IsInjected"] & ~scored["IsAnomaly"],"Case"]="false_negative"
    scored.loc[scored["Case"]!="true_negative",["Date","Return","ZScore","IsInjected","IsAnomaly","Case"]].to_csv(ROOT/"results/tables/synthetic_cases.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,metric in zip(axes,["Precision","Recall","F1"]):
        sns.heatmap(grid.pivot(index="Window",columns="Threshold",values=metric),annot=True,fmt=".2f",vmin=0,vmax=1,cmap="Blues",ax=ax); ax.set_title(metric)
    fig.tight_layout(); fig.savefig(ROOT/"results/figures/synthetic_metrics.png",dpi=180,bbox_inches="tight"); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--refresh",action="store_true"); parser.add_argument("--skip-download",action="store_true"); args=parser.parse_args()
    with open(ROOT/"config.json") as f: c=json.load(f)
    for d in [ROOT/"data/raw",ROOT/"data/processed",ROOT/"results/figures",ROOT/"results/tables"]: d.mkdir(parents=True,exist_ok=True)
    if not args.skip_download: real_experiment(c,args.refresh)
    synthetic_experiment(c); print("Analysis complete. See results/.")
if __name__=="__main__": main()
