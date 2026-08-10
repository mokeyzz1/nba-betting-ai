"""
Daily NBA prediction pipeline.

Fetch odds -> build features -> predict -> grade yesterday -> update metrics.

FAILURES ARE LOUD. The previous version wrapped every step in one bare
`except Exception` that printed the error and returned normally, so the
process exited 0 no matter what broke. Combined with the version mismatch
below, it ran for months printing "Pipeline completed successfully" while the
grading step silently found no file and did nothing. Under launchd that looks
identical to a healthy system.

Now: each step reports its own outcome, prediction and grading filenames both
derive from config.MODEL_VERSION, and any hard failure exits non-zero so the
scheduler and the logs disagree with each other instead of quietly agreeing.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta

from src.evaluate.evaluate_predictions import evaluate_results
from src.features.get_odds import fetch_odds
from src.features.get_today_games_features import build_features
from src.monitor.rolling_accuracy import update_rolling_accuracy
from src.pipeline.fetch_actual_winners import fetch_actual_results
from src.prediction.predict_today_enhanced import run_predictions
from src.utils.config import MODEL_VERSION, predictions_path


def _step(name: str, fn, *args, required: bool = True, **kwargs):
    """Run one step. Required steps abort the run; optional ones are reported."""
    print(f"\n>>> {name}")
    try:
        result = fn(*args, **kwargs)
        print(f"    ok: {name}")
        return result
    except Exception as e:
        print(f"    FAILED: {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        if required:
            raise
        return None


def main() -> int:
    today = datetime.today()
    today_str = today.strftime("%Y-%m-%d")
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"NBA PREDICTION PIPELINE  |  {today_str}  |  model: {MODEL_VERSION}")
    print("=" * 70)

    # --- today: produce picks -------------------------------------------
    odds = _step("Fetch odds", fetch_odds)

    # No games is a normal condition (offseason, All-Star break, a quiet
    # Monday), not a failure. Distinguish it from a real fault, or the
    # difference gets lost in a stack trace and the logs stop being readable.
    n_games = 0 if odds is None else len(odds)
    if n_games == 0:
        print(f"\n>>> No NBA games scheduled for {today_str}. Nothing to predict.")
        print("    (This is expected out of season. Not an error.)")
        return 0

    _step("Build features", build_features)
    _step("Run predictions", run_predictions)

    # The predictor must actually have written the file the grader will look
    # for tomorrow. Checking here turns a silent mismatch into a visible one.
    expected = predictions_path(today_str)
    if not expected.exists():
        print(f"\n!!! Predictions were not written to {expected.name}")
        print("    The grader will find nothing tomorrow. Check that the")
        print("    predictor derives its filename from config.MODEL_VERSION.")
        return 1
    print(f"\n    predictions written: {expected.name}")

    # --- yesterday: grade -----------------------------------------------
    # Optional: no games yesterday is normal, and should not fail the run.
    prev = predictions_path(yesterday)
    if prev.exists():
        _step(f"Fetch results for {yesterday}", fetch_actual_results,
              date=yesterday, model_version=MODEL_VERSION, required=False)
        _step(f"Grade {yesterday}", evaluate_results,
              date=yesterday, model_version=MODEL_VERSION, required=False)
        _step("Update rolling accuracy", update_rolling_accuracy,
              model_version=MODEL_VERSION, required=False)
    else:
        print(f"\n>>> Grading skipped: no predictions file for {yesterday}")

    print("\n" + "=" * 70)
    print("Pipeline finished.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
