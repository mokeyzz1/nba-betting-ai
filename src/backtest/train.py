"""
Train and persist the production model.

Trains the configuration the harness validated -- recency-weighted logistic
regression over pre-tipoff features -- and saves the model together with its
feature list and the metrics it achieved. Metadata travels with the model on
purpose: the previous generation of .pkl files in models/ carry no record of
what they were trained on or how they scored, so a claim like "10.2% ROI" in
a filename could never be checked against anything.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.backtest import dataset
from src.backtest.calibration import expected_calibration_error, recency_weights
from src.backtest.injury_features import INJURY_FEATURES, attach

BASE = Path(__file__).resolve().parent.parent.parent
MODELS = BASE / "models"
MODEL_PATH = MODELS / "nba_honest_v1.pkl"
META_PATH = MODELS / "nba_honest_v1.json"

HALF_LIFE_DAYS = 1460.0


def train(use_injuries: bool = True, verbose: bool = True):
    if verbose:
        print("Building dataset...")
    df = dataset.build(require_odds=False, verbose=verbose, include_recent=True)

    features = list(dataset.FEATURES)
    if use_injuries:
        df = attach(df, verbose=verbose)
        features += INJURY_FEATURES

    # Hold out the most recent season to report an honest out-of-sample score
    # alongside the shipped model, which is then refit on everything.
    seasons = sorted(df["season"].unique())
    holdout = seasons[-1]
    tr, te = df[df["season"] < holdout], df[df["season"] == holdout]

    def _fit(frame):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        w = recency_weights(frame["date"], HALF_LIFE_DAYS)
        m.fit(frame[features], frame["home_win"], logisticregression__sample_weight=w)
        return m

    probe = _fit(tr)
    p = probe.predict_proba(te[features])[:, 1]
    metrics = {
        "holdout_season": int(holdout),
        "holdout_games": int(len(te)),
        "accuracy": float(accuracy_score(te["home_win"], (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(te["home_win"], p)),
        "log_loss": float(log_loss(te["home_win"], p)),
        "ece": float(expected_calibration_error(te["home_win"].to_numpy(), p)),
    }
    if verbose:
        print(f"\nHold-out season {holdout} ({len(te):,} games):")
        for k in ("accuracy", "auc", "log_loss", "ece"):
            print(f"  {k:<10} {metrics[k]:.4f}")

    model = _fit(df)          # final model sees everything
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "name": "nba_honest_v1",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "n_train_games": int(len(df)),
        "train_span": [str(df["date"].min().date()), str(df["date"].max().date())],
        "half_life_days": HALF_LIFE_DAYS,
        "uses_odds_as_feature": False,
        "holdout_metrics": metrics,
        "notes": (
            "Recency-weighted logistic regression on pre-tipoff features only. "
            "Odds are deliberately excluded so model probability and market "
            "probability stay independent and edge means something."
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"\nSaved {MODEL_PATH.relative_to(BASE)}")
        print(f"Saved {META_PATH.relative_to(BASE)}  ({len(features)} features, "
              f"{len(df):,} training games)")
    return model, meta


if __name__ == "__main__":
    train()
