# AHRN Data Pipeline with Bloomberg Inputs, Dataset Construction and Feature Formulas

## 1. Data pipeline

AHRN is built around a monthly security-level panel. Bloomberg provides the security, market, fundamental, valuation and macro inputs; the project then aligns those inputs by ticker and month, constructs the return and liquidity variables used by the models, and adds historical S&P 500 membership.

The data flow is:

```text
Bloomberg security / market data
        |
        +---- daily OHLC history ------------------+
        |
        +---- security fundamentals / valuation ---+
        |
        +---- market / macro series ---------------+----> monthly panel
        |                                           |
S&P 500 historical constituent snapshots ----------+
                                                    |
                                                    v
                                      derived return / liquidity fields
                                                    |
                                                    v
                                      AHRN modelling dataset
```

The private Bloomberg-derived dataset is kept outside the public repository. The repository contains the code and this data dictionary so that the data preparation logic remains understandable without distributing the underlying data.

---

## 2. Dataset files

### `SP500_master_panel.csv`

This is the main modelling table. It contains **349,440 ticker-month observations and 31 columns**, covering **January 2000 through August 2026** across **1,092 securities**.

The table combines four types of information:

1. security identifiers and S&P 500 membership;
2. market and liquidity variables;
3. company fundamentals and valuation measures;
4. market-wide variables that are common to all securities on a given date.

The modelling code works from this table rather than pulling Bloomberg data during model execution.

### `SP500_macro_panel.csv`

This is the date-level version of the market and macro inputs. It contains **320 monthly observations** from January 2000 through August 2026:

```text
date
spx
vix
move
credit_spread
ust10y
curve_2y10y
fed_funds
risk_free_rate
```

The same macro variables are carried into the master panel so that every ticker-month observation has the corresponding market environment.

### `SP500_sectors_clean.csv`

Reference table containing:

```text
ticker
gics_sector
gics_industry
```

It provides the GICS classification used when the security-level panel is grouped or analysed by sector/industry.

### `OHLC.xlsx`

This workbook contains the daily Bloomberg historical price pulls used as the underlying market-price history. It has four worksheets:

```text
PX_OPEN
PX_HIGH
PX_LOW
PX_LAST
```

Each sheet contains the date series followed by the security columns. The historical window runs from **1 January 2000 to 31 August 2026**.

### `SPX as of *.xlsx`

These are historical S&P 500 constituent snapshots. Each snapshot contains:

```text
Ticker
Name
Weight
Shares
Price
```

The project uses the snapshots to reconstruct historical membership rather than treating today's S&P 500 constituents as if they had been members throughout the entire sample.

---

# 3. Bloomberg extraction

## 3.1 Daily OHLC history

The Bloomberg Excel formulas used in `OHLC.xlsx` are standard `BDH` historical-data pulls. The security identifier is stored in the first row of each security column.

### Open

```excel
_xll.BDH(B1,"PX_OPEN","1/1/2000","8/31/2026","Dir=V","Dts=S","Sort=A","Quote=C","QtTyp=Y","PER=D")
```

For subsequent security columns:

```excel
_xll.BDH(C1,"PX_OPEN","1/1/2000","8/31/2026","Dir=V","Dts=H","Sort=A","Quote=C","QtTyp=Y","PER=D")
```

### High

```excel
_xll.BDH(B1,"PX_HIGH","1/1/2000","8/31/2026","Dir=V","Dts=S","Sort=A","Quote=C","QtTyp=Y","PER=D")
```

### Low

```excel
_xll.BDH(B1,"PX_LOW","1/1/2000","8/31/2026","Dir=V","Dts=S","Sort=A","Quote=C","QtTyp=Y","PER=D")
```

### Last

```excel
_xll.BDH(B1,"PX_LAST","1/1/2000","8/31/2026","Dir=V","Dts=S","Sort=A","Quote=C","QtTyp=Y","PER=D")
```

For the remaining columns, the same formulas are filled across with the corresponding security cell and `Dts=H`.

The important extraction settings are:

| Setting | Value |
|---|---|
| Historical function | `BDH` |
| Start | `1/1/2000` |
| End | `8/31/2026` |
| Direction | `Dir=V` |
| Date handling | `Dts=S` / `Dts=H` depending on position in the sheet |
| Sort | `Sort=A` |
| Quote | `Quote=C` |
| Quote type | `QtTyp=Y` |
| Periodicity | `PER=D` |

These pulls provide the daily price history from which the monthly price/return and liquidity measures are built.

---

# 4. Master-panel fields

The master panel is organised as follows.

| Field | Purpose in AHRN |
|---|---|
| `date` | Monthly observation date |
| `ticker` | Security identifier |
| `total_return_index` | Total-return series used for return construction |
| `volume` | Trading volume |
| `volume_avg_30d` | 30-day average trading volume |
| `market_cap` | Equity market capitalisation |
| `bid` | Bid price |
| `ask` | Ask price |
| `eps` | Earnings per share |
| `dividend_yield` | Dividend yield |
| `short_interest_ratio` | Short-interest measure |
| `best_eps` | Best/consensus EPS measure |
| `px_to_sales` | Price-to-sales valuation measure |
| `price_to_book` | Price-to-book valuation measure |
| `oper_margin` | Operating margin |
| `gics_sector` | GICS sector classification |
| `gics_industry` | GICS industry classification |
| `spx` | S&P 500 index level |
| `vix` | VIX volatility index |
| `move` | MOVE rates-volatility index |
| `credit_spread` | Credit-spread environment |
| `ust10y` | 10-year U.S. Treasury yield |
| `curve_2y10y` | 2-year / 10-year yield-curve measure |
| `fed_funds` | Federal funds rate |
| `risk_free_rate` | Annualised risk-free rate |
| `monthly_return` | Monthly total return derived from the return index |
| `bid_ask_spread` | Absolute bid-ask spread |
| `dollar_volume` | Trading volume converted to a dollar-volume proxy |
| `momentum_12_1` | 12-month momentum excluding the most recent month |
| `excess_return` | Monthly return minus monthly risk-free rate |
| `in_sp500` | Historical S&P 500 membership flag |

The first 25 fields are the raw/assembled inputs to the panel. The last six are project-level derived fields or universe construction fields.

---

# 5. Derived fields

These calculations are performed after the Bloomberg inputs have been assembled into the monthly security panel.

## 5.1 Monthly return

The return index is converted to a one-month return within each security:

\[
R_{i,t}=\frac{TRI_{i,t}}{TRI_{i,t-1}}-1
\]

where:

- \(i\) is the security;
- \(t\) is the month;
- \(TRI\) is the total-return index.

Python implementation:

```python
df["monthly_return"] = (
    df.groupby("ticker")["total_return_index"].pct_change()
)
```

The calculation is performed after sorting by ticker and date.

## 5.2 Monthly risk-free rate

The Bloomberg risk-free rate is stored as an annualised percentage. The monthly decimal rate used by the return calculations is:

\[
r^f_{m}=\frac{r^f_{annual}}{12\times100}
\]

or equivalently:

\[
r^f_m=\frac{r^f_{annual}}{1200}
\]

## 5.3 Excess return

The security's monthly excess return is:

\[
ER_{i,t}=R_{i,t}-r^f_t
\]

This is the return variable used when the modelling pipeline evaluates performance relative to the risk-free rate.

## 5.4 Bid-ask spread

The absolute spread is:

\[
Spread_{i,t}=Ask_{i,t}-Bid_{i,t}
\]

Python:

```python
df["bid_ask_spread"] = df["ask"] - df["bid"]
```

## 5.5 Mid-price

Where a price is needed from the quote pair, the midpoint is:

\[
Mid_{i,t}=\frac{Bid_{i,t}+Ask_{i,t}}{2}
\]

## 5.6 Relative bid-ask spread

The feature-engineering notebook also uses the normalised spread:

\[
RelativeSpread_{i,t}
=\frac{Ask_{i,t}-Bid_{i,t}}
{(Ask_{i,t}+Bid_{i,t})/2}
\]

Python:

```python
mid = (df["bid"] + df["ask"]) / 2.0
df["relative_spread"] = (df["ask"] - df["bid"]) / mid
```

## 5.7 Dollar volume

The primary project construction is:

\[
DollarVolume_{i,t}=Volume_{i,t}\times TRI_{i,t}
\]

Python:

```python
df["dollar_volume"] = (
    df["volume"] * df["total_return_index"]
)
```

When the return-index price proxy is unavailable, the feature-engineering notebook uses the bid-ask midpoint:

\[
DollarVolume_{i,t}=Volume_{i,t}\times Mid_{i,t}
\]

## 5.8 Next-month target

The production pipeline predicts the following month's excess return:

\[
Target_{i,t}=R_{i,t+1}-r^f_{t+1}
\]

The target is only created when the next observation is the actual next calendar month. This prevents a missing month or a long gap in a security's history from being treated as a valid one-period-ahead observation.

The one-fold validation notebook performs the same operation using the already constructed `excess_return` field:

\[
TargetExcess_{i,t}=ER_{i,t+1}
\]

---

# 6. Factor and cross-sectional features

The one-fold validation pipeline converts several security characteristics into monthly cross-sectional signals.

## 6.1 Cross-sectional percentile rank

For a variable \(x\), each month's observations are ranked against the other securities observed in that month:

\[
Rank_{i,t}=PercentileRank(x_{i,t}\mid t)
\]

Implementation:

```python
df.groupby("date")[feature].rank(
    pct=True,
    method="average"
)
```

This puts variables with very different units onto a common cross-sectional scale.

## 6.2 Momentum

`momentum_12_1` represents the 12-month momentum signal excluding the most recent month. The modelling pipeline then converts it into a monthly cross-sectional percentile rank:

\[
MomentumSignal_{i,t}
=Rank_t(Momentum12\_1_{i,t})
\]

## 6.3 Value

The value composite combines valuation and earnings information:

\[
ValueSignal_{i,t}
=
mean\left(
1-Rank_t(P/S),
1-Rank_t(P/B),
Rank_t(EPS)
\right)
\]

Only fields available for a security at that date contribute to the composite.

## 6.4 Quality

The quality signal combines operating margin and best EPS:

\[
QualitySignal_{i,t}
=
mean\left(
Rank_t(OperatingMargin),
Rank_t(BestEPS)
\right)
\]

## 6.5 Liquidity

The liquidity signal combines company size and trading activity:

\[
LiquiditySignal_{i,t}
=
mean\left(
Rank_t(MarketCap),
Rank_t(DollarVolume)
\right)
\]

## 6.6 Log transformations

The feature pipeline uses signed/log transformations for variables with large scale differences or both positive and negative values.

For a non-negative variable such as market capitalisation:

\[
LogMCap=\log(1+MarketCap)
\]

For dollar volume:

\[
LogDollarVolume=\log(1+DollarVolume)
\]

For variables that can be negative, the signed-log transform is:

\[
SignedLog(x)=sign(x)\log(1+|x|)
\]

This is applied to fields such as EPS and valuation variables where appropriate.

---

# 7. S&P 500 membership construction

The historical S&P 500 workbooks are used to avoid a survivorship-biased universe.

Each snapshot is parsed from its filename. For example:

```text
SPX as of Aug 10 2026.xlsx
```

becomes:

```python
snapshot_date = pd.to_datetime(
    filename.replace("SPX as of ", "").replace(".xlsx", ""),
    format="%b %d %Y"
)
```

The constituent table provides:

```text
Ticker
Name
Weight
Shares
Price
```

The historical snapshots are then used to construct `in_sp500` for the security-month panel. This means the model's universe is determined by the S&P 500 composition relevant to the period being evaluated rather than by the current index membership.

---

# 8. Macro variables

The monthly macro table is deliberately kept separate as a date-level dataset before being attached to the security panel.

### S&P 500

`spx` provides the broad equity-market level used to describe the market environment.

### VIX

`vix` captures equity-market implied-volatility conditions.

### MOVE

`move` captures the corresponding volatility environment in U.S. Treasury rates.

### Credit spread

`credit_spread` provides the credit-market stress component used by the regime features.

### 10-year Treasury

`ust10y` provides the long-end U.S. Treasury yield.

### 2Y–10Y curve

`curve_2y10y` captures the shape of the Treasury curve used by the macro regime inputs.

### Federal funds

`fed_funds` captures the policy-rate environment.

### Risk-free rate

`risk_free_rate` is the annualised rate used to convert security returns into excess returns.

These variables are aligned to the monthly panel by date and are therefore identical across securities on the same month.

---

# 9. End-to-end construction

The final monthly observation for a security is produced in the following order:

```text
1. Pull Bloomberg security and market history
2. Pull monthly macro/risk variables
3. Build the historical S&P 500 constituent snapshots
4. Align all security observations by ticker and date
5. Calculate monthly return from the total-return index
6. Convert the annual risk-free rate to a monthly decimal rate
7. Calculate excess return
8. Calculate bid-ask spread and dollar volume
9. Attach S&P 500 membership
10. Construct momentum and cross-sectional factor features
11. Create the next-month excess-return target
12. Pass the resulting panel to the walk-forward modelling pipeline
```

The important separation is that **Bloomberg extraction happens before modelling**. The experiment code consumes a prepared panel; it does not depend on an active Bloomberg Terminal.

---

# 10. Reproducing the data pull

With Bloomberg Terminal and Excel access, the data can be rebuilt by reproducing the security-level historical pulls, the monthly macro series, and the S&P 500 snapshots described above.

For the historical OHLC component, the workbook formulas can be filled across the security universe using the `BDH` structures shown in Section 3. The remaining security and macro fields are then assembled into the master panel using the same ticker/date keys.

The project deliberately keeps this extraction stage separate from the modelling stage. The public repository therefore contains the extraction documentation and transformation formulas while the Bloomberg-derived dataset itself remains private.
