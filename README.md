# Markowitz Portfolio Optimization — Group Assignment

`group_project.py` runs the whole assignment: data download, the rolling
2016–2025 optimization for the four expected-return models (Historical Mean,
CAPM, FF3, FF5+Momentum), the 1/N and S&P 500 benchmarks, all performance
measures, tables and figures.

## Run

```
pip install yfinance pandas numpy scipy statsmodels matplotlib requests openpyxl
python group_project.py
```

The first run downloads data into `data/`; later runs reuse those CSVs, so the
results are reproducible and the script works offline. Delete a file in `data/`
to re-download it. The script also runs cell-by-cell (`#%%`) in Spyder / VS Code.

## Outputs

| Path | Contents |
| --- | --- |
| `outputs/results.xlsx` | Every table (performance, annual returns, by-period returns, formation stats, forecast quality, weights, expected stock returns, factor betas, universe by year, daily values) |
| `outputs/*.csv` | Main tables as CSV |
| `outputs/figures/` | fig1 growth of $1M · fig2 annual returns · fig3 expected vs realized · fig4 weights heatmap · fig5 expected stock returns 2025 · fig6 drawdowns |
| `outputs/run_log.txt` | Console output of the full run |

## Data sources

- Yahoo Finance (yfinance): daily adjusted closes for 40 stocks, `^SP500TR`
  (S&P 500 total return), `^IRX` (13-week T-bill yield, risk-free rate at formation)
- Kenneth French Data Library: daily 3-factor, 5-factor (2x3) and momentum files;
  monthly 3-factor file (for an unrounded daily RF)

## Troubleshooting: SSL "unable to get local issuer certificate"

On some Windows machines antivirus software intercepts HTTPS, and Python does
not trust its certificate. Build a CA bundle that includes the Windows root
certificates and point the libraries at it before running:

```
python -c "import ssl,certifi; open('ca_bundle.pem','w').write(open(certifi.where()).read()+''.join(ssl.DER_cert_to_PEM_cert(c) for s in ('ROOT','CA') for c,e,t in ssl.enum_certificates(s) if e=='x509_asn'))"
set CURL_CA_BUNDLE=%CD%\ca_bundle.pem
set REQUESTS_CA_BUNDLE=%CD%\ca_bundle.pem
python group_project.py
```

(PowerShell: `$env:CURL_CA_BUNDLE="$PWD\ca_bundle.pem"` etc.) This is only
needed for the first run, since data is cached afterwards.
