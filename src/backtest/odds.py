"""
American odds math. Single source of truth.

Every ROI number in this project must come through these functions.
The old evaluators (src/evaluate/evaluate_roi_v4_2.py,
src/evaluate/evaluate_all_with_results.py) applied decimal-odds formulas
to American odds, which is why performance/rolling_roi.csv reports
impossible values like -376%. Nothing here should be reimplemented inline.
"""
from __future__ import annotations

import numpy as np


def profit_per_unit(american_odds: float, won: bool) -> float:
    """Profit on a 1-unit stake. Losing a bet costs the full unit.

    >>> profit_per_unit(-150, True)
    0.6666666666666666
    >>> profit_per_unit(+130, True)
    1.3
    >>> profit_per_unit(-150, False)
    -1.0
    """
    if not won:
        return -1.0
    o = float(american_odds)
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def implied_prob(american_odds: float) -> float:
    """Implied win probability, vig included.

    This is the correct conversion. Note that train_model_v4_1.py and
    train_model_v4_2.py instead used `1 / home_odds`, which for -150 gives
    -0.0067 rather than 0.60 -- a ~100x scale error that only shows up at
    serve time, where the correct formula is used. Any model retrained from
    those scripts inherits that train/serve mismatch.
    """
    o = float(american_odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove the bookmaker's margin from a two-way market.

    Raw implied probabilities sum to >1; the excess is the vig. Normalising
    gives the book's actual estimate, which is the number a model has to beat.
    """
    pa, pb = implied_prob(odds_a), implied_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def vig(odds_a: float, odds_b: float) -> float:
    """Bookmaker margin on a two-way market, as a fraction. ~0.045 is typical."""
    return implied_prob(odds_a) + implied_prob(odds_b) - 1.0


def kelly_fraction(win_prob: float, american_odds: float, cap: float = 0.05) -> float:
    """Fraction of bankroll to stake, capped. Returns 0 when there is no edge."""
    b = profit_per_unit(american_odds, True)  # net decimal payout
    edge = win_prob * (b + 1.0) - 1.0
    if edge <= 0:
        return 0.0
    return float(min(edge / b, cap))


def settle(odds: np.ndarray, won: np.ndarray) -> np.ndarray:
    """Vectorised profit_per_unit over arrays."""
    odds = np.asarray(odds, dtype=float)
    won = np.asarray(won, dtype=bool)
    payout = np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))
    return np.where(won, payout, -1.0)


def implied_prob_array(odds: np.ndarray) -> np.ndarray:
    """Vectorised implied_prob over arrays."""
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, 100.0 / (odds + 100.0), np.abs(odds) / (np.abs(odds) + 100.0))
