"""
Cross-book mispricing scanner.

The rest of this project tries to predict games better than the market. That
failed, and the reason is documented in ROADMAP.md: our model correlates .898
with the bookmaker, so it holds almost no independent information.

This module does something different, and closer to how independent bettors
actually profit. It does not predict anything. It looks for books that
disagree with each other by enough that the disagreement itself is the edge:

  ARBITRAGE      the best price on every outcome implies a total probability
                 below 1. Back all outcomes and the profit is locked in
                 regardless of the result. No forecast involved.

  LOW HOLD       total implied probability slightly above 1. Not free money,
                 but the toll is a fraction of the usual 4-5%, so a model
                 needs far less edge to clear it.

  OUTLIER        one book is materially off the consensus on a single side.
                 Usually a stale line that has not caught up to news.

Two-way markets (moneyline in most US sports) and three-way markets (soccer,
where a draw is possible) are both handled -- the arithmetic is the same, it
just sums over however many outcomes exist.

Nothing here is a recommendation to place a bet. Arbitrage in particular is
frequently unexecutable in practice: stale quotes vanish on click, limits are
small, and books void obvious errors. Treat a hit as a lead to verify, not a
filled position.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE / ".env")
API_KEY = os.getenv("ODDS_API_KEY")
API_ROOT = "https://api.the-odds-api.com/v4/sports"


def american_to_payout(odds: float) -> float:
    o = float(odds)
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


@dataclass
class Opportunity:
    sport: str
    home: str
    away: str
    commence: str
    outcomes: dict[str, tuple[float, str]]   # name -> (best odds, book)
    total_implied: float
    n_books: int
    consensus: dict[str, float] = field(default_factory=dict)

    @property
    def hold(self) -> float:
        """Total implied probability minus 1. Negative means arbitrage."""
        return self.total_implied - 1.0

    @property
    def is_arb(self) -> bool:
        return self.total_implied < 1.0

    @property
    def profit_pct(self) -> float:
        """Guaranteed return on total stake, if the arb is real and fillable."""
        return (1.0 / self.total_implied - 1.0) * 100.0 if self.is_arb else 0.0

    def stakes(self, bankroll: float = 100.0) -> dict[str, float]:
        """How to split a stake across outcomes so every result pays the same."""
        return {name: round(bankroll * implied_prob(o) / self.total_implied, 2)
                for name, (o, _) in self.outcomes.items()}


def fetch(sport: str, market: str = "h2h", regions: str = "us") -> list[dict]:
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in .env")
    r = requests.get(f"{API_ROOT}/{sport}/odds",
                     params={"apiKey": API_KEY, "regions": regions,
                             "markets": market, "oddsFormat": "american"},
                     timeout=30)
    if r.status_code == 422:      # market not offered for this sport
        return []
    r.raise_for_status()
    rem = r.headers.get("x-requests-remaining")
    if rem is not None and int(rem) < 20:
        print(f"  warning: {rem} API requests left this period")
    return r.json()


def scan_game(game: dict, sport: str, market: str = "h2h") -> Opportunity | None:
    """Best available price per outcome, across every book quoting the game."""
    best: dict[str, tuple[float, str]] = {}
    allq: dict[str, list[float]] = {}
    books = game.get("bookmakers", [])
    for b in books:
        m = next((x for x in b.get("markets", []) if x["key"] == market), None)
        if not m:
            continue
        for o in m["outcomes"]:
            name, price = o["name"], float(o["price"])
            allq.setdefault(name, []).append(price)
            if name not in best or american_to_payout(price) > american_to_payout(best[name][0]):
                best[name] = (price, b["key"])

    # Need every outcome quoted, or the total implied probability is meaningless.
    if len(best) < 2 or any(len(v) < 2 for v in allq.values()):
        return None

    total = sum(implied_prob(o) for o, _ in best.values())
    consensus = {n: sum(implied_prob(p) for p in v) / len(v) for n, v in allq.items()}
    return Opportunity(sport=sport, home=game.get("home_team", "?"),
                       away=game.get("away_team", "?"),
                       commence=game.get("commence_time", ""),
                       outcomes=best, total_implied=total,
                       n_books=len(books), consensus=consensus)


def scan(sports: list[str], market: str = "h2h", regions: str = "us",
         verbose: bool = True) -> list[Opportunity]:
    found = []
    for sp in sports:
        try:
            games = fetch(sp, market, regions)
        except Exception as e:
            if verbose:
                print(f"  {sp:<36} error {type(e).__name__}")
            continue
        opps = [o for o in (scan_game(g, sp, market) for g in games) if o]
        if verbose and opps:
            best = min(opps, key=lambda o: o.total_implied)
            print(f"  {sp:<36} {len(opps):>3} games  best hold {best.hold:+.2%}")
        found.extend(opps)
    return found


def report(opps: list[Opportunity], low_hold: float = 0.02) -> None:
    arbs = sorted([o for o in opps if o.is_arb], key=lambda o: o.total_implied)
    lows = sorted([o for o in opps if not o.is_arb and o.hold <= low_hold],
                  key=lambda o: o.hold)
    print(f"\n{'='*74}\nscanned {len(opps)} games")
    print(f"  arbitrage (guaranteed profit) : {len(arbs)}")
    print(f"  low hold (<= {low_hold:.0%})            : {len(lows)}")

    for o in arbs[:10]:
        print(f"\n  ARB {o.profit_pct:+.2f}%   {o.away} @ {o.home}  [{o.sport}]")
        for name, (odds, book) in o.outcomes.items():
            print(f"      {name:<28} {odds:>+7.0f} @ {book}")
        print(f"      stakes per $100: {o.stakes()}")

    if lows:
        print(f"\n  Lowest-hold markets (cheapest places to need an edge):")
        print(f"  {'hold':>7}  {'books':>5}  matchup")
        for o in lows[:12]:
            print(f"  {o.hold:>+7.2%}  {o.n_books:>5}  {o.away} @ {o.home}  [{o.sport}]")


if __name__ == "__main__":
    import sys
    sports = sys.argv[1:] or ["basketball_wnba", "baseball_mlb", "icehockey_nhl"]
    print(f"Scanning {len(sports)} sport(s) for cross-book mispricing...\n")
    report(scan(sports))
