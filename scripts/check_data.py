from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "raw" / "SP500_master_panel.csv"
required = [
    "date", "ticker", "monthly_return", "risk_free_rate", "in_sp500",
    "volume_avg_30d", "bid_ask_spread", "market_cap",
]
if not path.exists():
    raise SystemExit(f"Missing dataset: {path}. Run scripts/download_data.py first.")
df = pd.read_csv(path, nrows=5)
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")
print(f"OK: {path.name} contains all required columns.")
