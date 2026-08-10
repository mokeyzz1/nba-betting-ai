"""
Calibration: making the probabilities mean what they say.

The walk-forward model is decently discriminative (AUC ~.73) but its
probabilities are inflated on the home side -- predicted 55% comes in at 45%,
predicted 81% at 75%. The cause is a regime change: home advantage fell from
~58% to ~55%, while training history is dominated by decades where it was
higher. The model faithfully learned an edge that no longer exists.

This matters more than accuracy. Accuracy only decides which side to back;
the probability decides whether the price is worth taking and how much to
stake. A model that says 55% when the truth is 45% will take bad prices
confidently and lose money even when it picks winners.

Two corrections, both temporally honest:

  1. Recency weighting -- exponential decay on sample age, so recent seasons
     dominate the fit without discarding the older ones outright.
  2. A post-hoc calibrator fitted on a held-out RECENT season, never on the
     test season. sklearn's CalibratedClassifierCV cross-fits on the training
     set, which mixes eras and cannot see the regime shift; fitting the
     calibrator on the most recent prior season can.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def recency_weights(dates: pd.Series, half_life_days: float = 1460.0) -> np.ndarray:
    """Exponential decay by age. Default half-life is four years.

    A game twice the half-life old counts a quarter as much as today's.
    Weighting rather than truncating keeps the long history contributing to
    stable coefficients while letting the current regime set the level.
    """
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age / float(half_life_days))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted and realised frequency, weighted by bin size."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return total / len(p)


def calibration_table(y: np.ndarray, p: np.ndarray,
                      edges=(0, .4, .5, .6, .7, 1.01)) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{lo:.0%}-{min(hi,1):.0%}", "n": int(m.sum()),
                     "predicted": p[m].mean(), "actual": y[m].mean(),
                     "gap": p[m].mean() - y[m].mean()})
    return pd.DataFrame(rows)


def _base_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                *, half_life_days: float | None = None,
                calibrate_on_last_season: bool = False,
                target: str = "home_win") -> np.ndarray:
    """Train and predict under one calibration strategy.

    When calibrate_on_last_season is set, the most recent season inside
    `train` is held out of model fitting and used only to fit an isotonic
    calibrator. The test season is never touched by either fit.
    """
    if calibrate_on_last_season:
        last = train["season"].max()
        fit_part = train[train["season"] < last]
        cal_part = train[train["season"] == last]
        if len(cal_part) < 300 or len(fit_part) < 3000:   # not enough to split
            fit_part, cal_part = train, None
    else:
        fit_part, cal_part = train, None

    model = _base_model()
    w = recency_weights(fit_part["date"], half_life_days) if half_life_days else None
    kw = {"logisticregression__sample_weight": w} if w is not None else {}
    model.fit(fit_part[features], fit_part[target], **kw)

    if cal_part is not None:
        raw_cal = model.predict_proba(cal_part[features])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(raw_cal, cal_part[target])
        return iso.predict(model.predict_proba(test[features])[:, 1])

    return model.predict_proba(test[features])[:, 1]
