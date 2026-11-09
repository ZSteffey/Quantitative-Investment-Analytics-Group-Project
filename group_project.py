# -*- coding: utf-8 -*-
"""
Group Assignment: Markowitz Portfolio Optimization and Performance Evaluation

Rolling 5-year mean-variance (max-Sharpe) portfolios, 2016-2025, using four
expected-return models:
    Method 1  Historical mean returns
    Method 2  CAPM
    Method 3  Fama-French 3-factor
    Method 4  Fama-French 5-factor + Momentum (our own model)
plus two benchmarks: 1/N equal weight and the S&P 500 (total return).

Builds on the class code:
    - mpt_python_inclass.py   -> yfinance download, annualized mu / Sigma, max_sharpe() with SLSQP
    - capm.py / capm exercise -> excess returns, beta regression, CAPM expected return
    - apt_python_inclass.py   -> Fama-French factors, sm.OLS betas, APT expected return

requires: pip install yfinance pandas numpy scipy statsmodels matplotlib requests openpyxl

Run top-to-bottom (python group_project.py) or cell-by-cell (#%%) in Spyder / VS Code.
Downloaded data is cached in ./data so later runs are reproducible and offline.
"""

#%%

# ---------------------------------------------------------------
# 0) Imports and settings
# ---------------------------------------------------------------

import os
import io
import zipfile
import warnings

import numpy as np                          # numerical computing (arrays, math)
import pandas as pd                         # tabular data handling (DataFrame)
import yfinance as yf                       # download stock data from Yahoo
import statsmodels.api as sm                # OLS regression
import requests                             # download Fama-French zip files
import matplotlib.pyplot as plt             # plotting
import matplotlib.ticker as mtick
from scipy.optimize import minimize         # Markowitz optimizer (SLSQP)

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.width", 160)

TRADING_DAYS = 252                 # annualization convention
FIRST_YEAR, LAST_YEAR = 2016, 2025 # investment (holding) years
WINDOW_YEARS = 5                   # rolling estimation window, calendar years
MAX_WEIGHT = 0.15                  # no stock above 15% at formation
INITIAL_INVESTMENT = 1_000_000     # $1,000,000 at the start of 2016
MIN_COVERAGE = 0.98                # stock must have returns on >= 98% of window days (= 5 full years)

PRICE_START = "2010-12-01"         # a few weeks before 2011 so the first 2011 return exists
PRICE_END   = "2026-01-01"         # yfinance 'end' is exclusive -> last day 2025-12-31

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

#%%

# ---------------------------------------------------------------
# 1) Stock universe (Part 1)
# ---------------------------------------------------------------
# Selection rule (decided with information available at the START of the
# sample, to limit hindsight / survivorship bias): large U.S. companies that
# were S&P 500 members in 2010, spread across all 11 GICS sectors, including
# names that later did badly (INTC, GE, F, FCX, SLB, BA, WBA-style laggards).
# We add three companies that IPO'd after 2011 (META, ABBV, PYPL) to show how
# listing dates are handled: their pre-IPO returns are MISSING (NaN), not zero,
# and they only enter the universe once they have five full years of returns.

sectors = {
    "AAPL": "Info Tech",   "MSFT": "Info Tech",   "IBM":  "Info Tech",
    "INTC": "Info Tech",   "CSCO": "Info Tech",   "ORCL": "Info Tech",
    "GOOGL": "Comm Svcs",  "T":    "Comm Svcs",   "VZ":   "Comm Svcs",
    "DIS":  "Comm Svcs",   "META": "Comm Svcs",
    "JPM":  "Financials",  "BAC":  "Financials",  "WFC":  "Financials",
    "BRK-B": "Financials", "PYPL": "Financials",
    "JNJ":  "Health Care", "PFE":  "Health Care", "MRK":  "Health Care",
    "UNH":  "Health Care", "AMGN": "Health Care", "ABBV": "Health Care",
    "PG":   "Cons Staples","KO":   "Cons Staples","PEP":  "Cons Staples",
    "WMT":  "Cons Staples",
    "AMZN": "Cons Discr",  "HD":   "Cons Discr",  "MCD":  "Cons Discr",
    "F":    "Cons Discr",
    "XOM":  "Energy",      "CVX":  "Energy",      "SLB":  "Energy",
    "GE":   "Industrials", "CAT":  "Industrials", "BA":   "Industrials",
    "MMM":  "Industrials",
    "FCX":  "Materials",
    "DUK":  "Utilities",
    "SPG":  "Real Estate",
}
tickers = list(sectors.keys())
SP500 = "^SP500TR"   # S&P 500 TOTAL RETURN index (dividends reinvested, comparable to adjusted closes)

print("Number of stocks:", len(tickers))

#%%

# ---------------------------------------------------------------
# 2) Download daily adjusted close prices (cached to CSV)
# ---------------------------------------------------------------
# If yfinance fails with an SSL "unable to get local issuer certificate" error,
# your antivirus/network is intercepting HTTPS. Setting CURL_CA_BUNDLE to a PEM
# file that includes your Windows root certificates fixes it (see README.md).

prices_file = os.path.join(DATA_DIR, "prices_adj_close.csv")

if os.path.exists(prices_file):
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    print("Loaded cached prices:", prices_file)
else:
    print("Downloading prices for", len(tickers), "stocks + S&P 500 ...")
    data = yf.download(tickers + [SP500], start=PRICE_START, end=PRICE_END,
                       auto_adjust=False, progress=True)
    prices = data["Adj Close"]                 # adjusted for splits & dividends
    prices = prices[tickers + [SP500]]         # keep a fixed column order
    prices.index.name = "Date"
    prices.to_csv(prices_file)
    print("Saved:", prices_file)

# daily simple returns: (P_t - P_{t-1}) / P_{t-1}
# pre-IPO prices are NaN -> pre-IPO returns stay NaN (missing, NOT zero)
all_rets = prices.pct_change(fill_method=None)
all_rets = all_rets.loc["2011-01-01":"2025-12-31"]

stock_rets = all_rets[tickers]
sp500_rets = all_rets[SP500]

# listing-date check
first_obs = stock_rets.apply(lambda s: s.first_valid_index())
print("\nStocks whose first return is after Jan 2011:")
print(first_obs[first_obs > pd.Timestamp("2011-01-31")])

# data-quality check: very large daily moves (bad adjustments, spin-offs)
big = stock_rets[stock_rets.abs() > 0.30].stack().dropna()
print("\nDaily returns larger than +/-30% (check these):")
print(big if len(big) else "none")

#%%

# ---------------------------------------------------------------
# 3) Fama-French daily factors from the Kenneth French Data Library
# ---------------------------------------------------------------
# The class code uses pandas_datareader's FamaFrenchReader; that package no
# longer works with pandas 3, so we read the same CSV zip files directly.

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

def read_french(zip_name, monthly=False):
    """Download a Ken French CSV zip and return a DataFrame in DECIMALS.
    Daily files have YYYYMMDD dates; monthly files have YYYYMM rows (we skip
    the annual YYYY block at the bottom)."""
    r = requests.get(FRENCH_URL + zip_name, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode("latin-1")

    date_len, date_fmt = (6, "%Y%m") if monthly else (8, "%Y%m%d")
    header, rows = None, []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts[0]) == date_len and parts[0].isdigit():   # data row
            rows.append(parts)
        elif len(parts) > 1 and not rows:                      # last comma line before data = header
            header = parts
        elif rows and not parts[0].isdigit():                  # end of the first data block
            break
    df = pd.DataFrame(rows, columns=["Date"] + header[1:])
    df["Date"] = pd.to_datetime(df["Date"], format=date_fmt)
    df = df.set_index("Date").astype(float) / 100.0         # percent -> decimals
    return df

ff_file = os.path.join(DATA_DIR, "ff_factors_daily.csv")

if os.path.exists(ff_file):
    ff = pd.read_csv(ff_file, index_col=0, parse_dates=True)
    print("Loaded cached factors:", ff_file)
else:
    ff3 = read_french("F-F_Research_Data_Factors_daily_CSV.zip")          # Mkt-RF, SMB, HML, RF
    ff5 = read_french("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")    # + RMW, CMA
    mom = read_french("F-F_Momentum_Factor_daily_CSV.zip")                # Mom
    ff3m = read_french("F-F_Research_Data_Factors_CSV.zip", monthly=True) # monthly RF

    # The daily file rounds RF to 0.01% per day (only 0, 1 or 2 bp), which is
    # up to ~1.3% per year of error. We rebuild daily RF from the monthly
    # 1-month T-bill return, spread evenly (compounded) over the month's trading days.
    month = ff3.index.to_period("M")
    n_days = pd.Series(1, index=ff3.index).groupby(month).transform("sum")
    rf_month = ff3m["RF"].reindex(month.to_timestamp()).values
    rf_daily = pd.Series((1 + rf_month) ** (1 / n_days.values) - 1, index=ff3.index)

    ff = pd.DataFrame({
        "MKT_RF": ff3["Mkt-RF"],
        "SMB":    ff3["SMB"],       # 3-factor SMB / HML (Method 3)
        "HML":    ff3["HML"],
        "RF":     rf_daily,         # daily 1-month T-bill return (from monthly RF)
        "SMB5":   ff5["SMB"],       # 5-factor versions (Method 4 uses the 5-factor set as published)
        "HML5":   ff5["HML"],
        "RMW":    ff5["RMW"],
        "CMA":    ff5["CMA"],
        "MOM":    mom.iloc[:, 0],
    })
    ff = ff.loc["2010-01-01":"2025-12-31"]
    ff.index.name = "Date"
    ff.to_csv(ff_file)
    print("Saved:", ff_file)

# 13-week U.S. T-bill yield (^IRX, annualized %), used as the risk-free rate at formation
tbill_file = os.path.join(DATA_DIR, "tbill_13w_yield.csv")
if os.path.exists(tbill_file):
    tbill = pd.read_csv(tbill_file, index_col=0, parse_dates=True).iloc[:, 0]
else:
    tbill = yf.download("^IRX", start=PRICE_START, end=PRICE_END,
                        auto_adjust=False, progress=False)["Close"].iloc[:, 0] / 100.0
    tbill.name = "IRX"
    tbill.index.name = "Date"
    tbill.to_frame().to_csv(tbill_file)
    print("Saved:", tbill_file)

# align trading dates: keep days present in BOTH Yahoo and French data
common = stock_rets.index.intersection(ff.index)
dropped = len(stock_rets) - len(common)
stock_rets = stock_rets.loc[common]
sp500_rets = sp500_rets.loc[common]
ff = ff.loc[common]
print(f"\nAligned sample: {common[0].date()} to {common[-1].date()}, "
      f"{len(common)} trading days ({dropped} Yahoo days not in French data)")

FACTORS = {
    "CAPM":    ["MKT_RF"],
    "FF3":     ["MKT_RF", "SMB", "HML"],
    "FF5+MOM": ["MKT_RF", "SMB5", "HML5", "RMW", "CMA", "MOM"],
}

#%%

# ---------------------------------------------------------------
# 4) Helper functions
# ---------------------------------------------------------------

def estimation_window(year):
    """Daily stock returns and factors for the five calendar years before `year`."""
    start, end = f"{year - WINDOW_YEARS}-01-01", f"{year - 1}-12-31"
    return stock_rets.loc[start:end], ff.loc[start:end]


def eligible_stocks(win_rets):
    """Stocks with (essentially) five full years of returns in the window."""
    counts = win_rets.count()
    return list(counts[counts >= MIN_COVERAGE * len(win_rets)].index)


def factor_betas(excess, factors):
    """OLS of each stock's daily excess return on the factors (like apt_python_inclass.py).
    Returns a DataFrame: rows = stocks, columns = const (alpha) + factor betas."""
    X = sm.add_constant(factors)
    out = {}
    for t in excess.columns:
        model = sm.OLS(excess[t], X, missing="drop").fit()
        out[t] = model.params
    return pd.DataFrame(out).T


def max_sharpe(mu, Sigma, r_f, max_w=MAX_WEIGHT):
    """Tangency portfolio: max (w'mu - r_f)/sqrt(w'Sigma w)
    s.t. sum(w) = 1, 0 <= w_i <= max_w   (long-only, no leverage, 15% cap).
    Same SLSQP set-up as max_sharpe() in mpt_python_inclass.py, plus the cap.
    We try a few starting points and keep the best solution."""
    N = len(mu)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bnds = [(0, max_w)] * N
    obj = lambda w: -(w @ (mu - r_f)) / np.sqrt(w @ Sigma @ w)   # negative Sharpe

    # starting points: equal weights, and the 7 best names by Sharpe / by excess return
    k = int(np.ceil(1 / max_w))
    starts = [np.ones(N) / N]
    for score in [(mu - r_f) / np.sqrt(np.diag(Sigma)), mu - r_f]:
        x0 = np.zeros(N)
        x0[np.argsort(-score)[:k]] = 1.0 / k
        starts.append(x0)

    best = None
    for x0 in starts:
        res = minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Optimizer failed")
    w = np.clip(best.x, 0, max_w)
    return w / w.sum()


def buy_and_hold(weights, hold_rets, start_value):
    """Value path of a portfolio bought at the start of the year and NOT rebalanced.
    Each position grows with its own cumulative return."""
    growth = (1 + hold_rets.fillna(0.0)).cumprod()   # value of $1 in each stock
    return start_value * (growth @ weights)


def max_drawdown(values):
    return (values / values.cummax() - 1).min()

#%%

# ---------------------------------------------------------------
# 5) Rolling optimization and backtest, 2016-2025 (Parts 2 and 3)
# ---------------------------------------------------------------

METHODS = ["Historical Mean", "CAPM", "FF3", "FF5+MOM"]
STRATS  = METHODS + ["Equal Weight", "S&P 500"]

value_paths = {s: [] for s in STRATS}             # daily portfolio values
current_value = {s: INITIAL_INVESTMENT for s in STRATS}

weights_rows   = []   # long table: year, method, ticker, weight
formation_rows = []   # expected vs realized, per year and method
mu_rows        = []   # stock-level expected returns per model + realized
beta_rows      = []   # factor loadings per stock-year (for the appendix)
universe_rows  = []

for year in range(FIRST_YEAR, LAST_YEAR + 1):

    # ----- 5a) estimation window and eligible universe (same for all methods) -----
    win_rets, win_ff = estimation_window(year)
    elig = eligible_stocks(win_rets)
    excluded = sorted(set(tickers) - set(elig))

    # one common estimation sample: days where every eligible stock has a return
    R = win_rets[elig].dropna()
    F = win_ff.loc[R.index]
    excess = R.subtract(F["RF"], axis=0)                       # R_i - R_f

    # ----- 5b) common inputs -----
    Sigma = R.cov().values * TRADING_DAYS                      # annualized sample covariance

    # Risk-free rate at formation: the 13-week T-bill yield on the last trading
    # day before the holding year. It is observable at formation (no look-ahead),
    # default-free, and it is the rate an investor could actually earn going
    # forward. The 5-year average RF would be stale (e.g. ~1% for 2018-2022
    # while T-bills yielded ~4.3% at the start of 2023).
    rf_form = tbill.loc[:f"{year - 1}-12-31"].dropna().iloc[-1]

    # ----- 5c) expected returns, four models (annualized) -----
    mu = {}

    # Method 1: historical mean daily return x 252
    mu["Historical Mean"] = R.mean() * TRADING_DAYS

    # Methods 2-4: E[r_i] = r_f + sum_k beta_ik * lambda_k, lambda_k = mean factor return x 252
    # (the regression intercept / alpha is NOT included)
    for name, cols in FACTORS.items():
        b = factor_betas(excess, F[cols])
        lambdas = F[cols].mean() * TRADING_DAYS
        mu[name] = rf_form + b[cols] @ lambdas
        b = b.rename(columns={"const": "alpha_daily"})
        b["model"], b["year"] = name, year
        beta_rows.append(b.rename_axis("ticker").reset_index())

    # ----- 5d) holding-year returns (out of sample) -----
    hold = stock_rets.loc[str(year), elig]
    realized_stock = (1 + hold.fillna(0)).prod() - 1

    for name in METHODS:
        for t in elig:
            mu_rows.append({"year": year, "model": name, "ticker": t,
                            "expected": mu[name][t], "realized": realized_stock[t]})

    # ----- 5e) optimize and hold -----
    port_weights = {}
    for name in METHODS:
        m = mu[name].values
        w = max_sharpe(m, Sigma, rf_form)
        port_weights[name] = w
        exp_ret = w @ m
        exp_vol = np.sqrt(w @ Sigma @ w)

        start_v = current_value[name]
        path = buy_and_hold(w, hold, start_v)
        value_paths[name].append(path)
        current_value[name] = path.iloc[-1]

        port_daily = path.pct_change()
        port_daily.iloc[0] = path.iloc[0] / start_v - 1
        formation_rows.append({
            "year": year, "method": name, "n_eligible": len(elig),
            "n_positive": int((w > 1e-4).sum()), "max_weight": w.max(),
            "rf_formation": rf_form,
            "expected_return": exp_ret, "expected_vol": exp_vol,
            "expected_sharpe": (exp_ret - rf_form) / exp_vol,
            "realized_return": path.iloc[-1] / start_v - 1,
            "realized_vol": port_daily.std() * np.sqrt(TRADING_DAYS),
        })
        for t, wt in zip(elig, w):
            weights_rows.append({"year": year, "method": name, "ticker": t, "weight": wt})

    # Benchmark 1: equal weight (1/N) on the same eligible universe
    w_eq = np.ones(len(elig)) / len(elig)
    path = buy_and_hold(w_eq, hold, current_value["Equal Weight"])
    value_paths["Equal Weight"].append(path)
    current_value["Equal Weight"] = path.iloc[-1]

    # Benchmark 2: S&P 500 total return index
    sp_hold = sp500_rets.loc[str(year)]
    path = current_value["S&P 500"] * (1 + sp_hold).cumprod()
    value_paths["S&P 500"].append(path)
    current_value["S&P 500"] = path.iloc[-1]

    universe_rows.append({"year": year,
                          "window": f"{year - WINDOW_YEARS}-{year - 1}",
                          "estimation_days": len(R),
                          "rf_formation": rf_form,
                          "n_eligible": len(elig),
                          "excluded": ", ".join(excluded)})
    print(f"{year}: window {year - WINDOW_YEARS}-{year - 1}, {len(elig)} eligible, "
          f"rf = {rf_form:.2%}, excluded: {excluded}")

# stitch the yearly paths together
values = pd.DataFrame({s: pd.concat(value_paths[s]) for s in STRATS})
start_row = pd.DataFrame(INITIAL_INVESTMENT, index=[pd.Timestamp("2015-12-31")], columns=STRATS)
values = pd.concat([start_row, values])
daily = values.pct_change().iloc[1:]

formation = pd.DataFrame(formation_rows)
weights_long = pd.DataFrame(weights_rows)
mu_long = pd.DataFrame(mu_rows)
betas_long = pd.concat(beta_rows, ignore_index=True)
universe = pd.DataFrame(universe_rows)

#%%

# ---------------------------------------------------------------
# 6) Performance evaluation, 2016-2025 (Part 4)
# ---------------------------------------------------------------
# Conventions:
#   annualized return  = geometric (CAGR) over the 10 years
#   annualized vol     = std(daily return) x sqrt(252)
#   Sharpe             = mean daily excess return x 252 / annualized vol
#   beta, Jensen alpha = OLS of daily portfolio excess return on daily
#                        S&P 500 (total return) excess return; alpha x 252
#   Treynor            = mean daily excess return x 252 / beta

rf_daily = ff.loc[daily.index, "RF"]
mkt_excess = daily["S&P 500"] - rf_daily
n_years = LAST_YEAR - FIRST_YEAR + 1

perf_rows = []
for s in STRATS:
    r = daily[s]
    ex = r - rf_daily
    reg = sm.OLS(ex, sm.add_constant(mkt_excess.rename("MKT"))).fit()
    beta = reg.params["MKT"]
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    ann_ex = ex.mean() * TRADING_DAYS
    end_val = values[s].iloc[-1]
    perf_rows.append({
        "Strategy": s,
        "Cumulative return": end_val / INITIAL_INVESTMENT - 1,
        "Annualized return": (end_val / INITIAL_INVESTMENT) ** (1 / n_years) - 1,
        "Annualized volatility": ann_vol,
        "Sharpe ratio": ann_ex / ann_vol,
        "Treynor ratio": ann_ex / beta,
        "Jensen's alpha": reg.params["const"] * TRADING_DAYS,
        "Alpha t-stat": reg.tvalues["const"],
        "Beta": beta,
        "Max drawdown": max_drawdown(values[s]),
        "Ending value": end_val,
    })
perf = pd.DataFrame(perf_rows).set_index("Strategy")
perf.loc["S&P 500", ["Jensen's alpha", "Alpha t-stat"]] = [0.0, np.nan]   # regressed on itself

# calendar-year returns for every strategy
year_end = values.groupby(values.index.year).last()
annual_returns = year_end.pct_change().loc[FIRST_YEAR:LAST_YEAR]
annual_returns.index.name = "year"

# sub-period summary (Part 5, Q4): market environments
periods = {
    "2016-2019 (pre-COVID bull)": (2016, 2019),
    "2020 (COVID)":                (2020, 2020),
    "2021 (recovery)":             (2021, 2021),
    "2022-2023 (rate hikes)":      (2022, 2023),
    "2024-2025 (AI rally)":        (2024, 2025),
}
period_rows = {}
for label, (a, b) in periods.items():
    g = (1 + annual_returns.loc[a:b]).prod() ** (1 / (b - a + 1)) - 1
    period_rows[label] = g
period_perf = pd.DataFrame(period_rows).T

print("\n=== Overall performance, 2016-2025 ===")
fmt = perf.copy()
for c in ["Cumulative return", "Annualized return", "Annualized volatility",
          "Jensen's alpha", "Max drawdown"]:
    fmt[c] = (fmt[c] * 100).map("{:.2f}%".format)
fmt["Treynor ratio"] = fmt["Treynor ratio"].map("{:.4f}".format)
fmt["Ending value"] = fmt["Ending value"].map("${:,.0f}".format)
print(fmt.round(3).to_string())

print("\n=== Calendar-year returns (%) ===")
print((annual_returns * 100).round(2).to_string())

print("\n=== Annualized return by market environment (%) ===")
print((period_perf * 100).round(2).to_string())

#%%

# ---------------------------------------------------------------
# 7) Formation statistics, weights, and model comparisons
# ---------------------------------------------------------------

weights_wide = (weights_long.pivot_table(index=["method", "ticker"], columns="year",
                                         values="weight", fill_value=0.0)
                .reindex(METHODS, level=0))

print("\n=== Expected vs realized portfolio returns (%) ===")
er = formation.pivot(index="year", columns="method", values="expected_return")[METHODS]
rr = formation.pivot(index="year", columns="method", values="realized_return")[METHODS]
ev = formation.pivot(index="year", columns="method", values="expected_vol")[METHODS]
npos = formation.pivot(index="year", columns="method", values="n_positive")[METHODS]
print(pd.concat({"Expected": er * 100, "Realized": rr * 100}, axis=1).round(1).to_string())
print("\nNumber of stocks with positive weight:")
print(npos.to_string())

# forecast summary: average expected vs average realized, bias and accuracy
fc_rows = []
for name in METHODS:
    f = formation[formation.method == name]
    gap = f.realized_return - f.expected_return
    s = mu_long[mu_long.model == name]
    ic = s.groupby("year").apply(lambda g: g["expected"].corr(g["realized"], method="spearman"))
    fc_rows.append({
        "Method": name,
        "Avg expected port. return": f.expected_return.mean(),
        "Avg realized port. return": f.realized_return.mean(),
        "Avg (realized - expected)": gap.mean(),
        "RMSE (port.)": np.sqrt((gap ** 2).mean()),
        "Avg expected vol": f.expected_vol.mean(),
        "Avg realized vol": f.realized_vol.mean(),
        "Avg # stocks held": f.n_positive.mean(),
        "Stock-level rank IC": ic.mean(),
        "Years IC > 0": int((ic > 0).sum()),
    })
forecast_summary = pd.DataFrame(fc_rows).set_index("Method")
print("\n=== Forecast quality ===")
print(forecast_summary.round(3).to_string())

# how different are the expected returns across models? (Part 5, Q3)
mu_wide = mu_long.pivot_table(index=["year", "ticker"], columns="model", values="expected")[METHODS]
mu_corr = mu_wide.groupby(level="year").corr().groupby(level=1).mean().loc[METHODS, METHODS]
mu_disp = pd.DataFrame({
    "Mean E[r]": mu_wide.groupby(level="year").mean().mean(),
    "Cross-sectional std of E[r]": mu_wide.groupby(level="year").std().mean(),
    "Min E[r]": mu_wide.groupby(level="year").min().mean(),
    "Max E[r]": mu_wide.groupby(level="year").max().mean(),
}).loc[METHODS]
print("\n=== Expected stock returns: average cross-sectional correlation between models ===")
print(mu_corr.round(2).to_string())
print("\n=== Expected stock returns: average dispersion (annualized) ===")
print(mu_disp.round(3).to_string())

# allocation overlap: sum_i min(w_a, w_b), averaged over years (1 = identical portfolios)
overlap = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
W = weights_long.pivot_table(index=["year", "ticker"], columns="method", values="weight")[METHODS]
for a in METHODS:
    for b in METHODS:
        overlap.loc[a, b] = np.minimum(W[a], W[b]).groupby(level="year").sum().mean()
print("\n=== Average portfolio overlap (sum of min weights) ===")
print(overlap.round(2).to_string())

# average realized exposures of each strategy (FF5 + MOM regression on daily returns)
expo_cols = FACTORS["FF5+MOM"]
expo_rows = {}
for s in STRATS:
    ex = daily[s] - rf_daily
    reg = sm.OLS(ex, sm.add_constant(ff.loc[daily.index, expo_cols])).fit()
    row = reg.params.copy()
    row["const"] = row["const"] * TRADING_DAYS
    row["R2"] = reg.rsquared
    expo_rows[s] = row
exposures = pd.DataFrame(expo_rows).T.rename(columns={"const": "alpha (ann.)"})
print("\n=== Realized factor exposures, 2016-2025 (FF5 + MOM regression) ===")
print(exposures.round(3).to_string())

#%%

# ---------------------------------------------------------------
# 8) Save tables
# ---------------------------------------------------------------

xlsx = os.path.join(OUT_DIR, "results.xlsx")
with pd.ExcelWriter(xlsx) as xw:
    perf.to_excel(xw, sheet_name="Performance")
    annual_returns.to_excel(xw, sheet_name="Annual returns")
    period_perf.to_excel(xw, sheet_name="By environment")
    formation.to_excel(xw, sheet_name="Formation stats", index=False)
    forecast_summary.to_excel(xw, sheet_name="Forecast quality")
    weights_wide.to_excel(xw, sheet_name="Weights")
    mu_wide.to_excel(xw, sheet_name="Expected stock returns")
    mu_corr.to_excel(xw, sheet_name="ER correlation")
    mu_disp.to_excel(xw, sheet_name="ER dispersion")
    overlap.to_excel(xw, sheet_name="Weight overlap")
    exposures.to_excel(xw, sheet_name="Realized exposures")
    betas_long.to_excel(xw, sheet_name="Factor betas", index=False)
    universe.to_excel(xw, sheet_name="Universe", index=False)
    values.to_excel(xw, sheet_name="Daily values")

perf.to_csv(os.path.join(OUT_DIR, "performance.csv"))
annual_returns.to_csv(os.path.join(OUT_DIR, "annual_returns.csv"))
formation.to_csv(os.path.join(OUT_DIR, "formation_stats.csv"), index=False)
weights_wide.to_csv(os.path.join(OUT_DIR, "weights.csv"))
mu_wide.to_csv(os.path.join(OUT_DIR, "expected_stock_returns.csv"))
values.to_csv(os.path.join(OUT_DIR, "daily_portfolio_values.csv"))
print("\nSaved tables to", OUT_DIR)

#%%

# ---------------------------------------------------------------
# 9) Figures
# ---------------------------------------------------------------
# Fixed colour per strategy (same in every chart). Optimized portfolios use
# the first four categorical slots; S&P 500 is a neutral dashed benchmark.

COLORS = {
    "Historical Mean": "#2a78d6",   # blue
    "CAPM":            "#eb6834",   # orange
    "FF3":             "#1baf7a",   # aqua
    "FF5+MOM":         "#6250d6",   # violet
    "Equal Weight":    "#e87ba4",   # magenta
    "S&P 500":         "#52514e",   # neutral gray (benchmark)
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ---- Figure 1: growth of $1,000,000 ----
fig, ax = plt.subplots(figsize=(10, 5.5))
for s in STRATS:
    ax.plot(values.index, values[s] / 1e6, color=COLORS[s], lw=2 if s != "S&P 500" else 1.8,
            ls="--" if s == "S&P 500" else "-", label=s)
# direct labels at the right edge, nudged apart so they don't collide
ends = (values.iloc[-1] / 1e6).sort_values()
ypos, min_gap = [], (ends.max() - ends.min()) * 0.09
for v in ends.values:
    ypos.append(max(v, ypos[-1] + min_gap) if ypos else v)
for (s, v), y in zip(ends.items(), ypos):
    ax.text(values.index[-1] + pd.Timedelta(days=40), y, f"{s}  ${v:.2f}M",
            va="center", fontsize=9, color=INK)
ax.set_ylabel("Portfolio value ($ millions)")
ax.set_title("Growth of $1,000,000, 2016–2025 (annual rebalancing)", loc="left", color=INK)
ax.legend(loc="upper left", ncol=2)
ax.set_xlim(values.index[0], values.index[-1])
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
fig.savefig(os.path.join(FIG_DIR, "fig1_growth_of_1m.png"))
plt.close(fig)

# ---- Figure 2: calendar-year returns ----
fig, ax = plt.subplots(figsize=(11, 5))
years = annual_returns.index.values
bw = 0.13
for i, s in enumerate(STRATS):
    ax.bar(years + (i - 2.5) * bw, annual_returns[s] * 100, width=bw * 0.9,
           color=COLORS[s], label=s, edgecolor="white", linewidth=0.5)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(years)
ax.set_ylabel("Calendar-year return (%)")
ax.set_title("Annual returns by strategy", loc="left", color=INK)
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG_DIR, "fig2_annual_returns.png"))
plt.close(fig)

# ---- Figure 3: expected vs realized portfolio returns ----
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, name in zip(axes.flat, METHODS):
    f = formation[formation.method == name]
    ax.bar(f.year - 0.2, f.expected_return * 100, width=0.38, color=COLORS[name],
           alpha=0.35, label="Expected at formation")
    ax.bar(f.year + 0.2, f.realized_return * 100, width=0.38, color=COLORS[name],
           label="Realized")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(name, loc="left", color=INK)
    ax.set_xticks(f.year)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="lower left")
for ax in axes[:, 0]:
    ax.set_ylabel("Portfolio return (%)")
fig.suptitle("Expected vs realized portfolio returns", x=0.02, ha="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_expected_vs_realized.png"))
plt.close(fig)

# ---- Figure 4: weights heatmaps ----
held = weights_long.groupby("ticker").weight.max()
show = [t for t in tickers if held.get(t, 0) > 1e-4]
fig, axes = plt.subplots(1, 4, figsize=(13, 8.5), sharey=True)
for ax, name in zip(axes, METHODS):
    M = (weights_long[weights_long.method == name]
         .pivot(index="ticker", columns="year", values="weight")
         .reindex(index=show, columns=range(FIRST_YEAR, LAST_YEAR + 1)))
    im = ax.imshow(M.values * 100, cmap="Blues", vmin=0, vmax=15, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([str(y)[2:] for y in M.columns], fontsize=8)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show, fontsize=8)
    ax.set_title(name, color=INK)
    ax.grid(False)
    # mark not-yet-eligible cells
    for i, t in enumerate(show):
        for j, y in enumerate(M.columns):
            if np.isnan(M.values[i, j]):
                ax.text(j, i, "·", ha="center", va="center", color=MUTED, fontsize=8)
cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
cb.set_label("Weight at formation (%)")
fig.suptitle("Portfolio weights at the start of each year ('·' = not yet eligible)",
             x=0.02, ha="left", color=INK)
fig.savefig(os.path.join(FIG_DIR, "fig4_weights_heatmap.png"))
plt.close(fig)

# ---- Figure 5: expected stock returns by model, latest formation year ----
yr = LAST_YEAR
E = mu_wide.loc[yr].sort_values("Historical Mean", ascending=False) * 100
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(E))
for i, name in enumerate(METHODS):
    ax.bar(x + (i - 1.5) * 0.2, E[name], width=0.19, color=COLORS[name], label=name)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(E.index, rotation=90, fontsize=8)
ax.set_ylabel("Expected annual return (%)")
ax.set_title(f"Expected stock returns at the start of {yr} (window {yr-5}–{yr-1})",
             loc="left", color=INK)
ax.legend(ncol=4)
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(FIG_DIR, "fig5_expected_returns_2025.png"))
plt.close(fig)

# ---- Figure 6: drawdowns ----
fig, ax = plt.subplots(figsize=(10, 4))
for s in STRATS:
    dd = values[s] / values[s].cummax() - 1
    ax.plot(dd.index, dd * 100, color=COLORS[s], lw=1.4,
            ls="--" if s == "S&P 500" else "-", label=s)
ax.set_ylabel("Drawdown from peak (%)")
ax.set_title("Drawdowns", loc="left", color=INK)
ax.legend(ncol=3, loc="lower left")
ax.set_xlim(values.index[0], values.index[-1])
fig.savefig(os.path.join(FIG_DIR, "fig6_drawdowns.png"))
plt.close(fig)

print("Saved figures to", FIG_DIR)
