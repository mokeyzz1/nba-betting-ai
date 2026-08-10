"""
Walk-forward backtest against real closing prices.

Two rules make this different from the previous build_enhanced_* scripts:

1. The model never sees the future. For each test season, it is trained only
   on seasons that finished before it. No shuffled train_test_split, which
   every script in src/training/ used on time-ordered data.

2. The model is graded against odds it did not invent. The old harnesses
   generated the "market" price from the model's own probability plus noise,
   then bet where that noise exceeded a threshold -- which selects the games
   the model already had right. Here the price comes from
   data/nba_historical_odds.csv and the model has no say in it.

Deliberately, the model is trained WITHOUT odds as features. A model fed the
price mostly learns to restate it, and any apparent "edge" is then just noise
around the market. Excluding the price makes the comparison meaningful: the
model forms an independent opinion, and we ask whether it beats the book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.backtest import dataset
from src.backtest.odds import implied_prob_array, settle

MIN_TRAIN_SEASONS = 3


def _make_model():
    """Calibrated logistic regression on standardised features.

    Deliberately simple. With ~1000 games per season and a dozen features,
    a stronger learner mostly fits noise, and probability quality matters far
    more than raw accuracy when the output is used for betting decisions.
    """
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def walk_forward(df: pd.DataFrame, features: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Train on all prior seasons, predict the next. Returns per-game predictions."""
    features = features or dataset.FEATURES
    seasons = sorted(df["season"].unique())
    out = []

    for i, season in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if len(test) < 100:
            continue

        model = _make_model()
        model.fit(train[features], train["home_win"])

        t = test.copy()
        t["model_prob"] = model.predict_proba(test[features])[:, 1]
        out.append(t)

        if verbose:
            acc = accuracy_score(t["home_win"], (t["model_prob"] >= 0.5).astype(int))
            print(f"    season {season}: train={len(train):>6,}  test={len(test):>5,}  acc={acc:.3f}")

    return pd.concat(out, ignore_index=True)


def _market_baselines(r: pd.DataFrame) -> dict:
    """What you get for free, without any model."""
    fav_is_home = r["home_odds"] < r["away_odds"]
    fav_odds = np.where(fav_is_home, r["home_odds"], r["away_odds"])
    fav_won = np.where(fav_is_home, r["home_win"] == 1, r["home_win"] == 0)
    fav_profit = settle(fav_odds, fav_won)

    home_profit = settle(r["home_odds"], r["home_win"] == 1)

    return {
        "always_home_roi": home_profit.mean() * 100,
        "always_home_acc": (r["home_win"] == 1).mean(),
        "always_fav_roi": fav_profit.mean() * 100,
        "always_fav_acc": fav_won.mean(),
        "market_logloss": log_loss(r["home_win"], r["market_prob_fair"]),
        "avg_vig": (r["market_prob_home_raw"] + r["market_prob_away_raw"] - 1).mean(),
    }


def evaluate(r: pd.DataFrame, verbose: bool = True) -> dict:
    """Classification quality, then betting performance at several edge cutoffs."""
    r = r.copy()
    r["market_prob_home_raw"] = implied_prob_array(r["home_odds"])
    r["market_prob_away_raw"] = implied_prob_array(r["away_odds"])
    tot = r["market_prob_home_raw"] + r["market_prob_away_raw"]
    r["market_prob_fair"] = r["market_prob_home_raw"] / tot     # vig removed
    r["edge"] = r["model_prob"] - r["market_prob_fair"]

    acc = accuracy_score(r["home_win"], (r["model_prob"] >= 0.5).astype(int))
    ll = log_loss(r["home_win"], r["model_prob"])
    auc = roc_auc_score(r["home_win"], r["model_prob"])
    base = _market_baselines(r)

    if verbose:
        print(f"\n  {'':<26}{'model':>10}{'market':>10}")
        print(f"  {'-'*46}")
        print(f"  {'accuracy':<26}{acc:>10.3f}{base['always_fav_acc']:>10.3f}")
        print(f"  {'log loss (lower=better)':<26}{ll:>10.4f}{base['market_logloss']:>10.4f}")
        print(f"  {'AUC':<26}{auc:>10.3f}{'-':>10}")
        print(f"\n  games graded              : {len(r):,}")
        print(f"  average vig               : {base['avg_vig']:.2%}")
        print(f"  flat-bet every home team  : {base['always_home_roi']:+.2f}% ROI")
        print(f"  flat-bet every favourite  : {base['always_fav_roi']:+.2f}% ROI")

    # Betting simulation. Bet whichever side the model likes, when its
    # disagreement with the de-vigged market exceeds the threshold.
    rows = []
    for thresh in (0.00, 0.02, 0.03, 0.05, 0.07, 0.10):
        bet_home = r["edge"] > thresh
        bet_away = r["edge"] < -thresh
        mask = bet_home | bet_away
        n = int(mask.sum())
        if n < 30:
            rows.append({"edge_threshold": thresh, "bets": n, "win_rate": np.nan,
                         "roi_pct": np.nan, "profit_units": np.nan, "roi_stderr_pct": np.nan})
            continue

        sel = r[mask]
        on_home = sel["edge"] > thresh
        odds = np.where(on_home, sel["home_odds"], sel["away_odds"])
        won = np.where(on_home, sel["home_win"] == 1, sel["home_win"] == 0)
        profit = settle(odds, won)

        rows.append({
            "edge_threshold": thresh,
            "bets": n,
            "win_rate": float(won.mean()),
            "roi_pct": float(profit.mean() * 100),
            "profit_units": float(profit.sum()),
            # How much of this is luck: 1 s.e. on the mean return.
            "roi_stderr_pct": float(profit.std(ddof=1) / np.sqrt(n) * 100),
        })

    sim = pd.DataFrame(rows)
    if verbose:
        print("\n  Betting simulation vs real closing prices")
        print("  " + "-" * 74)
        print(f"  {'edge':>6}{'bets':>8}{'win%':>9}{'ROI%':>9}{'±1se':>8}{'units':>10}   verdict")
        for _, x in sim.iterrows():
            if np.isnan(x["roi_pct"]):
                print(f"  {x['edge_threshold']:>6.0%}{int(x['bets']):>8}{'-':>9}{'too few bets':>26}")
                continue
            se = x["roi_stderr_pct"]
            v = ("profitable beyond noise" if x["roi_pct"] > 2 * se
                 else "positive, within noise" if x["roi_pct"] > 0
                 else "losing")
            print(f"  {x['edge_threshold']:>6.0%}{int(x['bets']):>8}{x['win_rate']:>9.1%}"
                  f"{x['roi_pct']:>+9.2f}{se:>8.2f}{x['profit_units']:>+10.1f}   {v}")

    return {"accuracy": acc, "log_loss": ll, "auc": auc, "sim": sim, **base}


def run(verbose: bool = True) -> dict:
    print("=" * 78)
    print("HONEST WALK-FORWARD BACKTEST")
    print("=" * 78)
    print("\n[1/3] Building dataset (pre-tipoff features only)")
    df = dataset.build(require_odds=True, verbose=verbose)

    print(f"\n[2/3] Walk-forward training ({df['season'].min()}-{df['season'].max()})")
    preds = walk_forward(df, verbose=verbose)

    print(f"\n[3/3] Grading against real closing odds")
    res = evaluate(preds, verbose=verbose)
    print("\n" + "=" * 78)
    return res


if __name__ == "__main__":
    run()
