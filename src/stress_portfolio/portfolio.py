"""The one authoritative ledger: cash, positions, and every transaction.

Price convention (checked against Yahoo data, e.g. APH 2:1 on 2024-06-12): `open`/`close` and
`dividends` are split-adjusted, so quantities are kept in split-adjusted share units and splits
need no quantity change. Never value on `adj_close`: it already includes dividends, which the
ledger credits as cash.
shortcut: quantities are only valid against the same price download they were sized on; once a
portfolio is carried across downloads (backtest milestone), store raw prices or apply splits.

Options live in `options` (contract ID -> contracts held) with their terms in `contracts`, never in `positions`,
so stock ordering and corporate actions never touch them. Option cash flows are contracts x multiplier x price.
Buying a put exchanges cash for an asset of the same value; later marks and closing cash flows give the P&L.
"""

from dataclasses import dataclass, field

import pandas as pd

TOLERANCE = 1e-6


@dataclass
class Portfolio:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    transactions: list[dict] = field(default_factory=list)
    options: dict[str, float] = field(default_factory=dict)
    contracts: dict[str, dict] = field(default_factory=dict)

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

    def add_contract(self, contract: str, terms: dict) -> None:
        """Register a contract's terms (underlying, option_type, strike, expiration, multiplier, exercise,
        settlement) before it is traded. The multiplier must be given explicitly, never assumed."""
        if terms.get("multiplier") is None or terms["multiplier"] <= 0:
            raise ValueError(f"{contract} needs a positive multiplier")
        self.contracts[contract] = dict(terms)

    def apply_option_fills(self, fills: pd.DataFrame, date) -> None:
        """Book option fills (see execution.execute_options) and their fees once. Long-only: no short options."""
        done = fills.loc[fills["status"].isin(["filled", "adjusted"]) & (fills["filled_qty"] != 0)]
        for f in done.itertuples():
            if f.contract not in self.contracts:
                raise KeyError(f"{f.contract} has no registered terms")
            mult = self.contracts[f.contract]["multiplier"]
            held = self.options.get(f.contract, 0.0)
            if held + f.filled_qty < -TOLERANCE:
                raise ValueError(f"sell of {f.contract} exceeds the {held} contracts held")
            self._record(date, "buy" if f.filled_qty > 0 else "sell", f.contract, f.filled_qty, f.fill_price,
                         -f.filled_qty * mult * f.fill_price)
            self._record(date, "commission", f.contract, 0.0, 0.0, -f.commission)
            self._set_option(f.contract, held + f.filled_qty)

    def expire_options(self, date, underlying_price: float) -> None:
        """Settle every held contract whose expiration is on or before `date`.

        Policy is to close puts before expiry, so this only runs when that failed (e.g. no quote). An
        out-of-the-money put expires worthless. An in-the-money SPY put settles in shares, which would leave a
        short SPY position the long-only mandate forbids; it is booked as exercise immediately covered at
        `underlying_price`, i.e. net intrinsic cash, and the note says so.
        shortcut: no exercise or assignment fee is charged; add one if a broker schedule is modeled.
        """
        for c, qty in list(self.options.items()):
            terms = self.contracts[c]
            if pd.Timestamp(terms["expiration"]) > pd.Timestamp(date):
                continue
            intrinsic = max(terms["strike"] - underlying_price, 0.0) if terms["option_type"] == "P" \
                else max(underlying_price - terms["strike"], 0.0)
            if intrinsic > 0:
                self._record(date, "exercise", c, -qty, intrinsic, qty * terms["multiplier"] * intrinsic,
                             f"exercised; short underlying covered at {underlying_price:.2f} (net intrinsic)")
            else:
                self._record(date, "expiry", c, -qty, 0.0, 0.0, "expired worthless")
            self._set_option(c, 0.0)

    def _set_option(self, contract: str, qty: float) -> None:
        if abs(qty) < TOLERANCE:
            self.options.pop(contract, None)
        else:
            self.options[contract] = qty

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

    def snapshot(self, date, prices: pd.Series, option_marks: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
        """Holdings valued at `prices` (stocks) and `option_marks` (per-share option prices, e.g. the bid-ask
        mid), plus a NAV row. Raises if the ledger does not reconcile or a held instrument has no mark.
        A zero option mark is valid (a worthless put); a negative or missing one is not."""
        self.reconcile()
        qty = pd.Series(self.positions, dtype=float)
        px = prices.reindex(qty.index)
        if px.isna().any() or (px <= 0).any():
            raise ValueError(f"no valid price on {date} for {px[px.isna() | (px <= 0)].index.tolist()}")
        holdings = pd.DataFrame({"ticker": qty.index, "quantity": qty.values, "price": px.values,
                                 "market_value": (qty * px).values})
        equity = float(holdings["market_value"].sum())
        derivatives = 0.0
        if self.options:
            oqty = pd.Series(self.options, dtype=float)
            marks = (option_marks if option_marks is not None else pd.Series(dtype=float)).reindex(oqty.index)
            if marks.isna().any() or (marks < 0).any():
                raise ValueError(f"no valid option mark on {date} for {marks[marks.isna() | (marks < 0)].index.tolist()}")
            mult = pd.Series({c: self.contracts[c]["multiplier"] for c in oqty.index}, dtype=float)
            opt = pd.DataFrame({"ticker": oqty.index, "quantity": oqty.values, "price": marks.values,
                                "market_value": (oqty * mult * marks).values})
            derivatives = float(opt["market_value"].sum())
            holdings = pd.concat([holdings, opt], ignore_index=True)
        nav = self.cash + equity + derivatives
        holdings["weight_of_nav"] = holdings["market_value"] / nav
        return holdings, {"date": date, "cash": self.cash, "equity_value": equity, "derivative_value": derivatives,
                          "nav": nav}

    def reconcile(self) -> None:
        """Cash must equal the sum of cash flows; positions must equal the sum of traded quantities."""
        ledger = self.ledger()
        if abs(self.cash - ledger["cash_flow"].sum()) > TOLERANCE:
            raise AssertionError("cash does not match the sum of ledger cash flows")
        moves = ledger.loc[ledger["type"].isin(["buy", "sell", "exercise", "expiry"])]
        traded = moves.groupby("instrument")["quantity"].sum()
        traded = traded[traded.abs() > TOLERANCE]
        if not traded.sort_index().equals(pd.Series(self.positions | self.options, dtype=float).sort_index()
                                          .rename_axis("instrument").rename("quantity")):
            raise AssertionError("positions do not match the sum of ledger trades")

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions,
                            columns=["date", "type", "instrument", "quantity", "price", "cash_flow", "note"])
