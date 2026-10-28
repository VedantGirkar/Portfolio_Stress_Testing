"""Small synthetic universe so every test runs offline."""

import numpy as np
import pandas as pd
import pytest

from stress_portfolio.config import Settings

SECTORS = ["Energy", "Financials", "Health Care", "Industrials", "Utilities"]


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(shortlist_size=20, core_size=5, base_slots_per_sector=2, max_per_sector=2, min_sectors=3,
                min_observations=300, min_median_dollar_volume=1e6, holding_counts=(5, 10), max_position_weight=0.25,
                data_dir=tmp_path / "data",
                runs_dir=tmp_path / "runs")
    return Settings(**{**base, **overrides})


def make_universe(n_per_sector: int = 6, sessions: int = 400, seed: int = 0):
    """Membership plus long bars: per-sector factor + noise so correlations differ by sector."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=sessions)
    rows, bars = [], []
    for si, sector in enumerate(SECTORS):
        factor = rng.normal(0, 0.01, sessions)
        for j in range(n_per_sector):
            t = f"{sector[:3].upper()}{j}"
            rows.append({"ticker": t, "provider_ticker": t, "company": f"{sector} Co {j}", "sector": sector,
                         "sub_industry": "x", "cik": si * 100 + j, "date_added": "2000-01-01",
                         "source": "test", "snapshot_date": "2025-07-01"})
            r = factor + rng.normal(0.0003 * j, 0.012, sessions)
            close = 50 * np.exp(np.cumsum(r))
            bars.append(pd.DataFrame({
                "date": dates, "ticker": t, "open": close, "high": close, "low": close, "close": close,
                "adj_close": close, "volume": 1e6 * (1 + j), "dividends": 0.0, "splits": 0.0}))
    return pd.DataFrame(rows), pd.concat(bars, ignore_index=True)


@pytest.fixture
def universe():
    return make_universe()
