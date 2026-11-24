# Point-in-time policy

## Timestamps

| Timestamp | Meaning | Column |
|---|---|---|
| price date | session date of a bar | daily index |
| return period | the month over which `ret` / `ret_fwd` accrues | `month`, `return_date` |
| fiscal period end | accounting period end | `fund_fiscal_period_end` |
| publication date | first public release (observed, or **assumed** from a lag) | `fund_publication_date` |
| vintage date | when this particular (possibly restated) version became available | `fund_vintage_date` |
| availability | `available_at = vintage_date` | `fund_available_at` |
| formation | close of the last session of month t | panel `date` |
| execution eligibility | first session after formation (next open in the backtester) | `exec_eligible` |

`fund_availability_source` records whether availability was **observed** or assumed from a
lag (`assumed_lag_days`). An assumed lag on revised database values is **not** proof of
point-in-time correctness: such databases often overwrite the original numbers with
restatements.

## As-of join (`panel.asof_join_fundamentals`)

For formation date t and ticker i, the join considers only records with `available_at ≤ t`.
It takes the latest fiscal period among them, and the latest vintage of that period that is
available by t. A restatement published after t is never used at t. Values are never
backfilled to dates before their availability. Fiscal periods more than 550 days old are
treated as stale. `assert_point_in_time` raises if any row uses a fundamental before its
availability. Tests cover original versus restated vintages, no backfill, staleness and
mutation of future data.

## Universe and survivorship

* **Synthetic mode**: late listings and delistings are simulated. A stock is in the universe
  at t only if it has a bar at the formation close t. Universe membership is never filtered on
  future completeness. Delisting returns are embedded in the final close and enter `ret_fwd`.
* **Real price-only mode**: the 127 tickers in `configs/real_universe.csv` are *current* large
  caps with *current* sector labels. This is survivorship-biased by construction: firms that
  failed or were acquired are absent. yfinance did not return data for MMC and BK. For EA and
  EQR it returned only 2026 bars, so they drop out before the hold-out cut. None were replaced.

## Corporate actions and units

* Prices are split- and dividend-adjusted (yfinance `auto_adjust=True`; synthetic prices are
  total-return-like). The backtester uses the same adjusted convention and does not model
  dividend cash events.
* Returns are decimal. French percent files are divided by 100 exactly once; a second
  conversion raises `UnitError`.
* One currency per run. `normalise_currency` converts GBp (pence) to GBP and refuses mixed
  currencies without FX data.
* An incomplete final month (data ending before the month's last business day) is dropped.

## User-supplied historical fundamentals

Use `data.validate_fundamentals` on a CSV with `ticker, fiscal_period_end, currency`, plus
`publication_date` (preferred) or `assumed_lag_days`, an optional `vintage_date`, and the value
columns `book_equity, net_income, gross_profit, total_assets, total_debt`. No university
database access is assumed. Without such data, historical value and quality research on real
stocks is **not run**, and the outputs say so.
