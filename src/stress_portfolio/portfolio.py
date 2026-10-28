"""The one authoritative ledger: cash, positions, and every transaction.

Price convention (checked against Yahoo data, e.g. APH 2:1 on 2024-06-12): `open`/`close` and
`dividends` are split-adjusted, so quantities are kept in split-adjusted share units and splits
need no quantity change. Never value on `adj_close`: it already includes dividends, which the
ledger credits as cash.
shortcut: quantities are only valid against the same price download they were sized on; once a
portfolio is carried across downloads (backtest milestone), store raw prices or apply splits.
"""

from dataclasses import dataclass, field

import pandas as pd

TOLERANCE = 1e-6


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    transactions: list[dict] = field(default_factory=list)

    def _record(self, date, kind: str, instrument: str, quantity: float, price: float, cash_flow: float,
                note: str = "") -> None:
        self.cash += cash_flow
        self.transactions.append({"date": date, "type": kind, "instrument": instrument, "quantity": quantity,
                                  "price": price, "cash_flow": cash_flow, "note": note})

    def deposit(self, date, amount: float) -> None:
        self._record(date, "deposit", "CASH", 0.0, 1.0, amount)

    def apply_fills(self, fills: pd.DataFrame, date) -> None:
        """Book each filled trade and its commission exactly once. Long-only: no selling below zero."""
        done = fills.loc[fills["status"].isin(["filled", "adjusted"]) & (fills["filled_qty"] != 0)]
        for f in done.itertuples():
            held = self.positions.get(f.ticker, 0.0)
            new = held + f.filled_qty
            if new < -TOLERANCE:
                raise ValueError(f"sell of {f.ticker} exceeds the {held} shares held")
            self._record(date, "buy" if f.filled_qty > 0 else "sell", f.ticker, f.filled_qty, f.fill_price,
                         -f.filled_qty * f.fill_price)
            self._record(date, "commission", f.ticker, 0.0, 0.0, -f.commission)
            if abs(new) < TOLERANCE:
                self.positions.pop(f.ticker, None)
            else:
                self.positions[f.ticker] = new

    def apply_corporate_actions(self, date, dividends: pd.Series, splits: pd.Series) -> None:
        """Credit dividends (per split-adjusted share) for shares held at the prior close.

        Credited on the ex-date rather than the pay date. Splits are logged only: prices are split-adjusted.
        """
        for t, qty in list(self.positions.items()):
            div = dividends.get(t, 0.0)
            if pd.notna(div) and div > 0:
                self._record(date, "dividend", t, 0.0, div, qty * div, "credited on ex-date")
            ratio = splits.get(t, 0.0)
            if pd.notna(ratio) and ratio > 0:
                self._record(date, "split", t, 0.0, ratio, 0.0, "already reflected in split-adjusted prices")

    def snapshot(self, date, prices: pd.Series) -> tuple[pd.DataFrame, dict]:
        """Holdings valued at `prices`, plus a NAV row. Raises if the ledger does not reconcile."""
        self.reconcile()
        qty = pd.Series(self.positions, dtype=float)
        px = prices.reindex(qty.index)
        if px.isna().any() or (px <= 0).any():
            raise ValueError(f"no valid price on {date} for {px[px.isna() | (px <= 0)].index.tolist()}")
        holdings = pd.DataFrame({"ticker": qty.index, "quantity": qty.values, "price": px.values,
                                 "market_value": (qty * px).values})
        equity = float(holdings["market_value"].sum())
        nav = self.cash + equity
        holdings["weight_of_nav"] = holdings["market_value"] / nav
        return holdings, {"date": date, "cash": self.cash, "equity_value": equity, "nav": nav}

    def reconcile(self) -> None:
        """Cash must equal the sum of cash flows; positions must equal the sum of traded quantities."""
        ledger = self.ledger()
        if abs(self.cash - ledger["cash_flow"].sum()) > TOLERANCE:
            raise AssertionError("cash does not match the sum of ledger cash flows")
        traded = ledger.loc[ledger["type"].isin(["buy", "sell"])].groupby("instrument")["quantity"].sum()
        traded = traded[traded.abs() > TOLERANCE]
        if not traded.sort_index().equals(pd.Series(self.positions, dtype=float).sort_index()
                                          .rename_axis("instrument").rename("quantity")):
            raise AssertionError("positions do not match the sum of ledger trades")

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions,
                            columns=["date", "type", "instrument", "quantity", "price", "cash_flow", "note"])
