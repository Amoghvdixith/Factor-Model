import numpy as np
import pandas as pd
import pytest

from factors.data import (
    SchemaError,
    UnitError,
    assert_decimal,
    monthly_returns_from_daily,
    normalise_currency,
    parse_french_csv,
    percent_to_decimal,
    rename_25_portfolios,
    validate_fundamentals,
)

FRENCH = """This file was created by a test.
The 1-month TBill return is from Ibbotson.

,Mkt-RF,SMB,HML,RF
202001,  1.00, -0.50,  2.00, 0.10
202002, -99.99,  0.25, -1.00, 0.12
202003, -12.50,  3.00,  0.00, 0.11

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2020,  5.00,  1.00, 2.00, 1.20
"""


def test_percent_to_decimal_exactly_once():
    df = parse_french_csv(FRENCH)
    assert df.attrs["units"] == "decimal"
    assert df.loc[pd.Period("2020-01", "M"), "MKT_RF"] == pytest.approx(0.01)
    assert df.loc[pd.Period("2020-03", "M"), "MKT_RF"] == pytest.approx(-0.125)
    with pytest.raises(UnitError):
        percent_to_decimal(df)


def test_missing_codes_become_nan_and_annual_block_ignored():
    df = parse_french_csv(FRENCH)
    assert np.isnan(df.loc[pd.Period("2020-02", "M"), "MKT_RF"])
    assert len(df) == 3  # the YYYY annual rows are not parsed as months
    assert list(df.columns) == ["MKT_RF", "SMB", "HML", "RF"]


def test_assert_decimal_flags_percent_data():
    with pytest.raises(UnitError):
        assert_decimal(pd.DataFrame({"a": [1.5, -2.0, 3.0, 0.8]}))
    assert_decimal(pd.DataFrame({"a": [0.015, -0.02, 0.03]}))


def test_rename_25_positional():
    df = pd.DataFrame(np.zeros((2, 25)), columns=[f"c{i}" for i in range(25)])
    out = rename_25_portfolios(df)
    assert out.columns[0] == "S1B1" and out.columns[4] == "S1B5" and out.columns[24] == "S5B5"
    with pytest.raises(SchemaError):
        rename_25_portfolios(df.iloc[:, :24])


def test_monthly_returns_need_previous_month_end_observation():
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    close = pd.DataFrame({"A": np.linspace(100, 120, len(idx)), "B": np.nan}, index=idx)
    close.loc["2021-02-15":, "B"] = 50.0
    m = monthly_returns_from_daily(close)
    assert np.isnan(m.loc[pd.Period("2021-02", "M"), "B"])  # not alive at Jan month end
    assert m.loc[pd.Period("2021-03", "M"), "B"] == pytest.approx(0.0)
    a = close["A"]
    exp = a.loc["2021-03-31"] / a.loc["2021-02-26"] - 1
    assert m.loc[pd.Period("2021-03", "M"), "A"] == pytest.approx(exp)


def _fund(**kw):
    base = {"ticker": ["X", "X"], "fiscal_period_end": ["2019-12-31", "2019-12-31"],
            "currency": ["USD", "USD"], "book_equity": [10.0, 11.0]}
    base.update(kw)
    return pd.DataFrame(base)


def test_fundamentals_observed_vs_assumed_lag():
    f = validate_fundamentals(pd.DataFrame({
        "ticker": ["X", "Y"], "fiscal_period_end": ["2019-12-31", "2019-12-31"],
        "currency": ["USD", "USD"], "publication_date": ["2020-03-01", None],
        "assumed_lag_days": [None, 120]}))
    src = dict(zip(f["ticker"], f["availability_source"], strict=True))
    assert src == {"X": "observed", "Y": "assumed_lag"}
    assert f.loc[f["ticker"] == "Y", "available_at"].iloc[0] == pd.Timestamp("2020-04-29")


def test_fundamentals_schema_errors():
    with pytest.raises(SchemaError):  # vintage before publication
        validate_fundamentals(_fund(publication_date=["2020-03-01", "2020-03-01"],
                                    vintage_date=["2020-01-01", "2020-06-01"]))
    with pytest.raises(SchemaError):  # no availability information
        validate_fundamentals(pd.DataFrame({"ticker": ["X"], "fiscal_period_end": ["2019-12-31"],
                                            "currency": ["USD"]}))
    with pytest.raises(SchemaError):  # duplicate vintage
        validate_fundamentals(_fund(publication_date=["2020-03-01", "2020-03-01"]))


def test_gbp_pence_normalisation_and_currency_mixing():
    df = pd.DataFrame({"px": [250.0, 2.5], "currency": ["GBp", "GBP"]})
    out = normalise_currency(df, ["px"])
    assert out["px"].tolist() == [2.5, 2.5] and set(out["currency"]) == {"GBP"}
    with pytest.raises(UnitError):
        normalise_currency(pd.DataFrame({"px": [1.0, 1.0], "currency": ["USD", "GBP"]}), ["px"])
