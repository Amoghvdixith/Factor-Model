# Backtester integration (project 02)

## Status

**Integrated through the public API**, version `backtester 0.1.0`. The source folder was not
modified. Its file listing and modification times were compared before and after this build:
all 314 files were unchanged.

## How it is installed

The backtester is a local package and is not on PyPI. It is deliberately **not** declared in
`pyproject.toml`, because an unpinned name could resolve to an unrelated PyPI project.
Install it from your own checkout:

```powershell
uv pip install --python .venv\Scripts\python.exe <path-to-your-Backtester-folder>
```

For this build, the backtester's `src/` and `pyproject.toml` were copied to a temporary folder
and installed from there as a regular (non-editable) wheel. This avoided writing `build/` or
`*.egg-info` files into the backtester folder. No Git URL or Git command was used.

## Adapter (`src/factors/backtester_adapter.py`)

* `TargetScheduleStrategy(backtester.Strategy)` returns `SignalEvent(direction=target_weight)`
  at the close of each formation date in a `{formation_date: weights}` schedule. The schedule
  is computed walk-forward from data available at each date.
* `PassThroughSizer(backtester.Sizer)` uses the direction as the target weight.
* The engine supplies everything else:
  - `PortfolioConstraints(max_position=5%, allow_short=False)`;
  - `CostModel.proportional(10 bps)`;
  - `BacktestConfig(execution_delay=1, fractional=False, cash_buffer=0.2%)`.

Conventions respected (not re-implemented):

| Convention | Effect here |
|---|---|
| signal at close t, fill at the next session's **open** | targets formed at the month-end close execute at the first session of the next month |
| complete target view per signal batch | names not in the target are sold |
| whole-share rounding, cash buffer, `resize` on insufficient cash | small residual cash (≈0.05% average) |
| adjusted prices, no dividend cash events | consistent with the panel |
| no bar → no fill; held positions stay marked at the last close | see the delisting note below |

## Target versus realised

`compare_targets` reports the following for each rebalance:

* Σ|target − realised| at the execution-day close;
* residual cash;
* realised maximum weight;
* weight stuck in untradeable names.

Average Σ|target − realised| was 4.5% in the fast demo (5.8% in research mode). It comes from:

* next-open execution and intraday drift;
* rounding and the cash buffer;
* **delisted holdings.**

The engine cannot sell a stock that has no further bars. A delisted synthetic stock therefore
stays as a frozen position, marked at its final close (which includes the delisting return).
By the end of the fast-mode OOS period, about 4.2% of equity was in such names. This is an
engine convention that the adapter reports but does not override. Modifying the backtester to
convert delisting proceeds to cash would need your approval.

## Limitation of the design

The backtester's `Strategy` sees only market data, not the portfolio. The optimiser's pre-trade
weights are therefore the *model* targets drifted by close-to-close returns, not the engine's
realised holdings. The engine's realised holdings differ by the gaps listed above. Turnover
limits are enforced on the model path. The executed turnover is in `results/tables/*_21_*`
and the engine's traded notional is in the tearsheet.
