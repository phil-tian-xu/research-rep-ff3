# Fama and French (1992) replication

Replicates **The Cross-Section of Expected Stock Returns**, *Journal of Finance*
47(2), 427–465, using the paper in `docs/raw_paper/`.

The target is the **1992 stock-level cross-sectional study**, including main
Tables I–VI and appendix Tables AI–AIV. The project directory's `ff3` name does
not change the target to the 1993 three-factor time-series paper. Kenneth
French's public factor/portfolio files are supplementary benchmarks; they do
not supply the stock-level panel or duplicate the original portfolio definitions.

## Run

```powershell
uv sync
uv run python main.py --skip-download
uv run pytest
```

The first command creates an independent environment. The second reuses the
dated raw extracts already prepared for this project. A complete run writes
`results/report.html`, figures, paper comparisons, regression slopes, sample
counts, a data audit, and a source manifest. The HTML report embeds its figures
and can be opened directly in a browser. Its CSV links refer to adjacent files.

For a new WRDS download, set `WRDS_USERNAME` and configure libpq's saved password
file (`%APPDATA%/postgresql/pgpass.conf` on Windows). No credentials are stored in
this repository:

```powershell
uv run python main.py --prepare-only
uv run python main.py --skip-download
```

An existing compatible cache from the momentum project can reduce downloads:

```powershell
uv run python main.py --reference-project ../JNS1993_VME_Momentum --prepare-only
```

This reads its original legacy CRSP/CCM extracts without importing, running, or
modifying that project's code. Reference checksums and provenance are retained.
Only early stock history, dated name classifications, paper-specific accounts,
and market indexes are fetched separately. Without this option, the downloader
can retrieve the entire panel directly from WRDS.

Useful options:

- `--skip-appendix`: main Tables I–VI only.
- `--tax-variable txditc --output-dir results/txditc_sensitivity`: an explicitly
  different book-equity sensitivity, keeping the primary run separate.
- `--refresh-data`: deliberately replace raw snapshots from WRDS; cannot be
  combined with `--reference-project`.

## Data and timing

- Raw inputs: `data/raw/`. Original source values are preserved. Source JSONs
  record SQL, retrieval dates or reference provenance, row counts, and SHA-256.
- Intermediate formation assignments, betas, and stock-months:
  `data/intermediate/txdb/` (or a separate tax-variable directory).
- Final outputs: `results/`. Proprietary extracts and large generated panels
  are excluded from Git. The small paper benchmark fixture is versioned in
  `config/paper_benchmarks.json`, separately from computed results.

Main tests use July 1963–December 1990 (330 months), fiscal year-ends in
calendar 1962–1989, June portfolio formation, December market equity for
valuation ratios, and June market equity for size. The appendix uses NYSE
stocks in January 1941–December 1990 (600 months) and December formation.

CRSP monthly returns are **legacy FIZ RET**, in decimals. Market equity is
`abs(prc) * shrout / 1000`, in USD millions. No separate delisting-return merge
or missing-return fill is applied. Formed constituents are followed regardless
of later exchange changes, using observed surviving stock returns each month.

The main formation sample requires common shares (10/11), an eligible exchange
(NYSE/AMEX/Nasdaq), December and June market values, 24–60 preceding monthly
return pairs, and accounting data. SIC 6000–6999 is excluded; historical annual
Compustat SIC is preferred to dated CRSP SIC. Missing SIC is excluded. Negative
or zero book equity and nonpositive assets/market equity are excluded.

Accounts use domestic USD Compustat records and the latest fiscal year-end in
calendar t−1. Book equity is **CEQ + TXDB** (balance-sheet deferred taxes), and
earnings are **IB + TXDI − DVP**. Missing tax and preferred-dividend add-ons are
zero; missing base amounts remain missing. `TXDITC` (deferred taxes **and
investment tax credits**) is available as a sensitivity, not silently substituted
for the paper's definition. These missing-value and item-mapping choices are
explicit implementation assumptions, since the paper does not specify every
modern Compustat edge case.

CCM links must be valid at formation and have LC/LU/LS link type with P/C
primary status. Overlaps prefer P, then LC/LU/LS, then the latest link start.
No additional two-year Compustat history screen is introduced.

## Estimation

Pre-ranking beta is the sum of slopes on the contemporaneous and one-month-lag
market return, with an intercept, estimated on up to 60 preceding calendar
months (minimum 24 valid pairs). Market lags are constructed before slicing
the window. Returns are raw, not excess returns. Main tests use CRSP's combined
market index; the appendix uses the NYSE-only MSIA index.

Main size breakpoints use all NYSE common stocks with June size data. Financial
firms and firms without accounts therefore still contribute to size cutoffs.
Size–beta portfolios use NYSE beta breakpoints within each size decile.
Size–book-to-market portfolios instead split the eligible stocks within each
size decile. Univariate B/M and E/P cutoffs use the full eligible sample. Extreme
deciles are split into two halves for the paper's 12-portfolio tables; negative
earnings form the extra E/P portfolio.

Post-ranking betas use the full sample of portfolio returns and are assigned
to constituent stocks. **This intentional ex-post estimation follows the
paper; it is not an out-of-sample forecasting or trading strategy.**

Fama–MacBeth regressions use monthly stock returns in **percent**, an intercept,
and the paper's explanatory variables. Table III ratios are winsorized at the
monthly 0.5/99.5 percentiles before logarithms. Reported t-statistics use the
conventional time-series standard error of monthly slopes, without HAC.
Monthly portfolio statistics are averaged through time; All margins weight
stocks directly, not portfolio averages.

The **descriptive NYSE market rows only** in Tables VI/AIV use log returns,
`100 * log(1 + R)`. This convention is inferred from the published statistics:
the 1941–1990 NYSE VW log mean is 0.931% versus the paper's 0.93%, while its
arithmetic mean is 1.022%; log standard deviations and t-statistics also match.
Both conventions are saved in separate CSVs. This does **not** transform market
returns in beta regressions, stock returns, or FM coefficients.

## Comparing results

Use `results/main/*_comparison.csv` and `results/appendix/*_comparison.csv` for
paper-versus-replication diagnostics (the stored `difference` is **replication
minus paper**). Published means are rounded. Modern CRSP, Compustat, and CCM
vintages cannot establish numerical identity with the original 1992 extracts.
Sample counts and each design assumption should be checked before interpreting
deviations as an implementation error.

French's public 100 size/B/M portfolios use **independent NYSE breakpoints**,
whereas Table V uses B/M sorts **within** size groups. Their universe and current
CIZ return definitions also differ, as does the library's later book-equity
definition (preferred-stock adjustments and investment tax credits). The public-data comparison is saved under
`results/french_benchmarks/`, restricted to the original 330 months, and labelled
as a comparison between different constructions. French missing-value codes
remain missing. Each comparison reports available-month counts and recomputes
our corresponding average using those same months; the paper comparison keeps
the original full sample.

Tests cover calendar-aligned beta estimation, minimum history, exclusion of
future observations, quantiles and ties, fiscal-year matching, July/June holding
assignment, missing returns, FM standard errors, stock-weighted All margins,
and a small synthetic end-to-end run of every table. Synthetic data remain in
the test temporary directory and never enter the real-data results.
