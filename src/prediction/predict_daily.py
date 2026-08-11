"""
Daily predictions from the backtest-validated model.

Replaces predict_today_enhanced.py, whose "Hybrid Elite" model ran with 6 of
its 10 features frozen at placeholder defaults because get_team_stats() never
supplied PIE, PIE_RANK or AST_TO.

Two properties this module is built around:

TRAIN/SERVE PARITY. Live features come from dataset.current_team_state(),
which reads the state that dataset.build() produced -- the same Elo and the
same rolling windows the model was trained on. Nothing is recomputed by hand.
The v4.2 model failed precisely here: training used `1 / home_odds` while
serving used the correct conversion, so features arrived ~100x off scale.

INDEPENDENT EDGE. The model never sees the price, so its probability and the
market's are independent estimates and their difference means something. Edge
is measured against the de-vigged consensus across books, and bets are priced
at the BEST available number rather than the first book returned.
"""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.backtest import dataset
from src.backtest.injury_features import INJURY_FEATURES, live_burden
from src.backtest.odds import kelly_fraction
from src.backtest.train import META_PATH, MODEL_PATH
from src.features.line_shopping import fetch, implied_prob, parse_all
from src.utils.config import MODEL_VERSION, predictions_path

# The odds feed and the box-score data spell a few teams differently.
TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Los Angeles Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}

MIN_EDGE = 0.03          # below this the number is inside model noise
BANKROLL = 1000.0
MAX_STAKE_FRACTION = 0.02


def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train it first:\n"
            f"    python -m src.backtest.train"
        )
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def _rest(state_row, today: pd.Timestamp) -> tuple[float, float]:
    """Days since a team's last game, and whether it is on a back-to-back.

    dataset.current_team_state() carries each team's last game date, so rest
    is derivable rather than assumed. Clipped at 10 to match dataset.py.
    """
    last = pd.Timestamp(state_row["date"])
    days = float((today.normalize() - last.normalize()).days)
    days = float(np.clip(days, 0, 10))
    return days, 1.0 if days <= 1 else 0.0


def _features_for(home: str, away: str, state: pd.DataFrame,
                  today: pd.Timestamp, burden: dict) -> dict | None:
    """Assemble one game's feature row from current team state.

    Every feature the model was trained on is computed here. Earlier this
    function hardcoded rest, back-to-backs and all three injury features to
    zero -- 5 of 15 frozen, which is the same defect that made the old Hybrid
    Elite model worthless. They are now derived from real inputs.
    """
    home_full = TEAM_ALIASES.get(home, home)
    away_full = TEAM_ALIASES.get(away, away)
    h = state[state["team"] == home_full]
    a = state[state["team"] == away_full]
    if h.empty or a.empty:
        return None
    h, a = h.iloc[0], a.iloc[0]

    h_rest, h_b2b = _rest(h, today)
    a_rest, a_b2b = _rest(a, today)

    f = {
        "elo_diff": h["elo"] - a["elo"],
        "home_elo": h["elo"],
        "away_elo": a["elo"],
        "win_pct_5_diff": h["win_pct_5"] - a["win_pct_5"],
        "win_pct_10_diff": h["win_pct_10"] - a["win_pct_10"],
        "margin_5_diff": h["margin_5"] - a["margin_5"],
        "margin_10_diff": h["margin_10"] - a["margin_10"],
        "off_form_diff": h["pts_10"] - a["opp_pts_10"],
        "def_form_diff": a["pts_10"] - h["opp_pts_10"],
        "season_win_pct_diff": h["season_win_pct"] - a["season_win_pct"],
        "rest_diff": h_rest - a_rest,
        "b2b_diff": h_b2b - a_b2b,
    }

    # Injury burden from today's published report. Sign convention matches
    # injury_features.attach(): positive means the away side is more depleted.
    hb = burden.get(home_full, {})
    ab = burden.get(away_full, {})
    f["inj_burden_diff"] = ab.get("inj_burden", 0.0) - hb.get("inj_burden", 0.0)
    f["inj_burden_out_diff"] = ab.get("inj_burden_out", 0.0) - hb.get("inj_burden_out", 0.0)
    f["inj_listed_diff"] = ab.get("inj_listed", 0.0) - hb.get("inj_listed", 0.0)
    return f


def run_predictions(verbose: bool = True) -> pd.DataFrame | None:
    model, meta = _load()
    features = meta["features"]

    games = parse_all(fetch("basketball_nba"))
    if not games:
        if verbose:
            print("No NBA games with odds right now.")
        return None

    if verbose:
        print(f"Building current team state from history...")
    state = dataset.current_team_state()
    today = pd.Timestamp(datetime.today().date())
    burden = live_burden(today, verbose=verbose)
    if not burden and verbose:
        print("  WARNING: no injury data today -- injury features will be neutral,\n"
              "           which understates absences. Predictions are degraded.")

    rows, skipped = [], []
    for g in games:
        f = _features_for(g.home_team, g.away_team, state, today, burden)
        if f is None:
            skipped.append(f"{g.away_team} @ {g.home_team}")
            continue

        quotes = list(g.quotes.values())
        hq = next((q for q in quotes if q.outcome == g.home_team), None)
        aq = next((q for q in quotes if q.outcome == g.away_team), None)
        if hq is None or aq is None:
            skipped.append(f"{g.away_team} @ {g.home_team} (incomplete prices)")
            continue

        fair_home = g.consensus_fair.get(g.home_team, implied_prob(hq.odds))
        rows.append({
            "hometeam": g.home_team, "awayteam": g.away_team,
            "commence_time": g.commence_time,
            "home_odds": hq.odds, "away_odds": aq.odds,
            "home_book": hq.book, "away_book": aq.book,
            "n_books": hq.n_books,
            "market_prob_home": fair_home,
            **f,
        })

    if not rows:
        if verbose:
            print(f"No predictable games. Skipped: {skipped}")
        return None

    df = pd.DataFrame(rows)

    # The guard the old predictor lacked. A feature that is constant across a
    # whole slate is a placeholder, not a measurement, and a model fed
    # placeholders still returns confident-looking probabilities.
    if len(df) > 2:
        frozen = [c for c in features if df[c].nunique(dropna=False) <= 1]
        # Genuinely-constant-by-nature features are exempt only when the whole
        # slate legitimately shares a value (e.g. nobody on any team is hurt).
        if len(frozen) > len(features) // 3:
            raise RuntimeError(
                f"{len(frozen)} of {len(features)} features are constant across "
                f"{len(df)} games: {frozen}\n"
                "That is placeholder data, not measurement. Fix the feature "
                "source before trusting these predictions."
            )
        if frozen and verbose:
            print(f"  note: {len(frozen)} feature(s) constant today: {frozen}")

    df["model_prob_home"] = model.predict_proba(df[features])[:, 1]
    df["edge_home"] = df["model_prob_home"] - df["market_prob_home"]

    df["pick"] = np.where(df["edge_home"] > 0, "HOME", "AWAY")
    df["pick_prob"] = np.where(df["pick"] == "HOME",
                               df["model_prob_home"], 1 - df["model_prob_home"])
    df["pick_odds"] = np.where(df["pick"] == "HOME", df["home_odds"], df["away_odds"])
    df["pick_book"] = np.where(df["pick"] == "HOME", df["home_book"], df["away_book"])
    df["edge"] = df["edge_home"].abs()
    df["bet"] = df["edge"] >= MIN_EDGE
    df["stake"] = [
        round(kelly_fraction(p, o, cap=MAX_STAKE_FRACTION) * BANKROLL, 2) if b else 0.0
        for p, o, b in zip(df["pick_prob"], df["pick_odds"], df["bet"])
    ]

    df = df.sort_values("edge", ascending=False)
    today = datetime.today().strftime("%Y-%m-%d")
    out = predictions_path(today, MODEL_VERSION)
    df.to_csv(out, index=False)

    if verbose:
        print(f"\n{len(df)} games | model: {meta['name']} | "
              f"holdout AUC {meta['holdout_metrics']['auc']:.3f}")
        cols = ["awayteam", "hometeam", "pick", "pick_prob", "market_prob_home",
                "edge", "pick_odds", "pick_book", "stake"]
        print(df[cols].to_string(index=False, formatters={
            "pick_prob": "{:.1%}".format, "market_prob_home": "{:.1%}".format,
            "edge": "{:+.1%}".format, "pick_odds": "{:+.0f}".format,
            "stake": "${:.2f}".format}))
        n = int(df["bet"].sum())
        print(f"\n{n} bet(s) clear the {MIN_EDGE:.0%} edge threshold.")
        if skipped:
            print(f"Skipped {len(skipped)}: {skipped}")
        print(f"Saved -> {out.name}")
    return df


if __name__ == "__main__":
    run_predictions()
