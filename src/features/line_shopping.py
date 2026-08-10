"""
Multi-book odds fetching and best-price selection.

The previous fetchers (src/features/get_odds.py, get_enhanced_odds.py) read
`bookmakers[0]` -- whichever book The Odds API happened to return first. Every
price the system ever recorded came from one arbitrary book.

That is the single most expensive line in the project. Sportsbooks disagree,
and taking the best available price is worth more than most modelling work,
because it pays on every bet regardless of how good the model is. Where a
model has to be right to earn, a better price earns automatically.

This module is sport-agnostic: basketball_nba in season, basketball_wnba now.
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

# Books that price sharply and move first. Their consensus is the better
# estimate of true probability; the soft books are where the value sits.
SHARP_BOOKS = {"pinnacle", "lowvig", "betonlineag", "bookmaker"}


def american_to_payout(odds: float) -> float:
    """Profit per 1 unit staked. Higher is always better for the bettor."""
    o = float(odds)
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


@dataclass
class Quote:
    """The best available price on one side, and who is offering it."""
    outcome: str
    odds: float
    book: str
    payout: float
    n_books: int
    worst_odds: float
    first_book_odds: float          # what the old bookmakers[0] code would have taken

    @property
    def gain_vs_first_book(self) -> float:
        """Extra profit per unit staked, versus taking the first book's price."""
        return self.payout - american_to_payout(self.first_book_odds)


@dataclass
class GameOdds:
    home_team: str
    away_team: str
    commence_time: str
    quotes: dict[str, Quote] = field(default_factory=dict)
    consensus_fair: dict[str, float] = field(default_factory=dict)
    sharp_fair: dict[str, float] = field(default_factory=dict)

    @property
    def best_line_vig(self) -> float:
        """Vig if you took the best price on BOTH sides across books.

        Often near zero, occasionally negative -- a negative value is an
        arbitrage: the books disagree enough that both sides can be backed
        for a guaranteed profit.
        """
        return sum(implied_prob(q.odds) for q in self.quotes.values()) - 1.0


def fetch(sport: str = "basketball_nba", markets: str = "h2h",
          regions: str = "us") -> list[dict]:
    """Raw API response. One request; check the quota headers before looping."""
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in .env")
    r = requests.get(
        f"{API_ROOT}/{sport}/odds",
        params={"apiKey": API_KEY, "regions": regions,
                "markets": markets, "oddsFormat": "american"},
        timeout=30,
    )
    if r.status_code == 401:
        raise RuntimeError("Odds API rejected the key (401)")
    if r.status_code == 429:
        raise RuntimeError("Odds API quota exhausted (429)")
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining")
    if remaining is not None and int(remaining) < 25:
        print(f"  warning: only {remaining} API requests remaining this period")
    return r.json()


def _devig(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()} if total else probs


def parse_game(game: dict, market: str = "h2h") -> GameOdds | None:
    """Collapse every book's quote for one game into best prices per side."""
    books = game.get("bookmakers", [])
    if not books:
        return None

    # outcome -> list of (odds, book_key)
    collected: dict[str, list[tuple[float, str]]] = {}
    first_book: dict[str, float] = {}
    sharp: dict[str, list[float]] = {}

    for b in books:
        m = next((x for x in b.get("markets", []) if x["key"] == market), None)
        if not m:
            continue
        for o in m["outcomes"]:
            name, price = o["name"], float(o["price"])
            collected.setdefault(name, []).append((price, b["key"]))
            first_book.setdefault(name, price)          # first book that quoted it
            if b["key"] in SHARP_BOOKS:
                sharp.setdefault(name, []).append(price)

    if not collected:
        return None

    quotes = {}
    for name, entries in collected.items():
        best_odds, best_book = max(entries, key=lambda e: american_to_payout(e[0]))
        worst_odds = min(entries, key=lambda e: american_to_payout(e[0]))[0]
        quotes[name] = Quote(
            outcome=name, odds=best_odds, book=best_book,
            payout=american_to_payout(best_odds), n_books=len(entries),
            worst_odds=worst_odds, first_book_odds=first_book[name],
        )

    # Fair probabilities: average the implied prob across all books, then devig.
    consensus = _devig({
        name: sum(implied_prob(p) for p, _ in entries) / len(entries)
        for name, entries in collected.items()
    })
    sharp_fair = _devig({
        name: sum(implied_prob(p) for p in prices) / len(prices)
        for name, prices in sharp.items()
    }) if len(sharp) >= 2 else {}

    return GameOdds(
        home_team=game["home_team"], away_team=game["away_team"],
        commence_time=game["commence_time"], quotes=quotes,
        consensus_fair=consensus, sharp_fair=sharp_fair,
    )


def parse_all(games: list[dict], market: str = "h2h") -> list[GameOdds]:
    return [g for g in (parse_game(x, market) for x in games) if g]


def shopping_report(parsed: list[GameOdds]) -> dict:
    """Quantify what best-price selection is worth versus the old approach.

    Reports the median, not the mean. A single longshot dominates the mean:
    moving a +590 underdog to +650 is worth 0.60 units on its own, which on a
    small slate drags the average far above what you would actually realise.
    The median is the number to plan around.
    """
    import statistics as _st

    gains, spreads, books_per, arbs = [], [], [], 0
    improved = 0
    for g in parsed:
        for q in g.quotes.values():
            gains.append(q.gain_vs_first_book)
            spreads.append(q.payout - american_to_payout(q.worst_odds))
            books_per.append(q.n_books)
            if q.gain_vs_first_book > 1e-9:
                improved += 1
        if g.best_line_vig < 0:
            arbs += 1
    n = len(gains) or 1
    return {
        "games": len(parsed),
        "quotes": len(gains),
        "avg_books_per_side": sum(books_per) / n,
        "median_gain_vs_first_book_pct": _st.median(gains) * 100 if gains else 0.0,
        "mean_gain_vs_first_book_pct": _st.mean(gains) * 100 if gains else 0.0,
        "median_best_vs_worst_pct": _st.median(spreads) * 100 if spreads else 0.0,
        "sides_improved_pct": improved / n * 100,
        "arbitrage_games": arbs,
    }


if __name__ == "__main__":
    import sys
    sport = sys.argv[1] if len(sys.argv) > 1 else "basketball_nba"
    print(f"Fetching {sport} ...")
    parsed = parse_all(fetch(sport))
    if not parsed:
        print("No games with odds right now (out of season?).")
        raise SystemExit(0)

    for g in parsed:
        print(f"\n{g.away_team} @ {g.home_team}")
        for q in g.quotes.values():
            print(f"  {q.outcome:<26} best {q.odds:>+7.0f} @ {q.book:<14}"
                  f" (first book {q.first_book_odds:>+7.0f}, {q.n_books} books)")
        print(f"  vig at best prices: {g.best_line_vig:+.2%}")

    r = shopping_report(parsed)
    print(f"\n{'-'*66}")
    print(f"games {r['games']} | quotes {r['quotes']} | avg {r['avg_books_per_side']:.1f} books/side")
    print(f"best vs first book (median): {r['median_gain_vs_first_book_pct']:+.2f}% per unit staked")
    print(f"best vs first book (mean)  : {r['mean_gain_vs_first_book_pct']:+.2f}%  (longshot-skewed)")
    print(f"best vs worst book (median): {r['median_best_vs_worst_pct']:+.2f}% per unit staked")
    print(f"sides where shopping helped: {r['sides_improved_pct']:.0f}%")
    print(f"arbitrage opportunities    : {r['arbitrage_games']}")
