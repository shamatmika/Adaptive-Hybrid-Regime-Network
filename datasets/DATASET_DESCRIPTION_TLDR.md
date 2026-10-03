# Dataset description

The supplied Bloomberg-derived archive contains four core data resources plus historical S&P 500 membership snapshots.

## Core files

| File | Observed size | Purpose in the supplied project |
|---|---:|---|
| `SP500_master_panel.csv` | 349,440 × 31 | Main security-month panel used by the AHRN implementation. |
| `SP500_macro_panel.csv` | 320 × 9 | Monthly market/macro variables. |
| `SP500_sectors_clean.csv` | 1,092 × 3 | Ticker-to-GICS sector/industry mapping. |
| `OHLC.xlsx` | workbook | OHLC data supplied with the Bloomberg pull. |
| `SPX as of Aug *.xlsx` | 27 workbooks | Historical August S&P 500 snapshot files used by the supplied notebook. |

## Master-panel columns

`date`, `ticker`, `total_return_index`, `volume`, `volume_avg_30d`, `market_cap`, `bid`, `ask`, `eps`, `dividend_yield`, `short_interest_ratio`, `best_eps`, `px_to_sales`, `price_to_book`, `oper_margin`, `gics_sector`, `gics_industry`, `spx`, `vix`, `move`, `credit_spread`, `ust10y`, `curve_2y10y`, `fed_funds`, `risk_free_rate`, `monthly_return`, `bid_ask_spread`, `dollar_volume`, `momentum_12_1`, `excess_return`, `in_sp500`.

The executable AHRN pipeline additionally requires the configured model features and uses the panel's monthly return, risk-free rate, liquidity variables, and constituent flag.

## Important provenance limitation

The supplied CSV/XLSX files establish the fields and observed structure, but they do **not** document the original Bloomberg field identifiers, terminal settings, timestamp conventions, or exact Excel extraction formulas. Those details should not be invented. The separate extraction guide records what can be established from the supplied materials and identifies the provenance items that still require the original Bloomberg pull instructions.
