# Pre-registered execution protocol

Primary test: A1 versus B1 on mean fold-level net annualised Sharpe, using Role-A exogenous transaction costs. A1 must win a majority of paired folds and pass the planned paired test at 10%. It must remain positive when each crisis fold is removed. The code reports these conditions; failure is the planned negative finding.

Role A is always bid/ask half-spread plus a volume/market-cap impact proxy. Role B applies only while A2–A4 train. Role C changes position sizes for A3–A4 only using regime-probability entropy. No model output enters Role-A cost.

All numerical parameters are configured in `configs/default.yaml`. Preserve the emitted `run_manifest.json`, fold logs, and configuration copy with every final result.
