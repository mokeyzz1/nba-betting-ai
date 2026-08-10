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
from src.evaluate.evaluate_roi import grade_all
from src.monitor.rolling_accuracy import update_rolling_accuracy
from src.pipeline.fetch_actual_winners import fetch_actual_results
from src.prediction.predict_daily import run_predictions
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
    # predict_daily fetches its own prices across all books and builds
    # features from dataset.current_team_state(), so the live feature code
    # and the backtest feature code are literally the same code.
    preds = _step("Predict today's games", run_predictions)

    # No games is a normal condition (offseason, All-Star break, a quiet
    # Monday), not a failure. Distinguish it from a real fault, or the
    # difference gets lost in a stack trace and the logs stop being readable.
    if preds is None or len(preds) == 0:
        print(f"\n>>> No NBA games with odds for {today_str}. Nothing to predict.")
        print("    (This is expected out of season. Not an error.)")
        return 0

    # The predictor must actually have written the file the grader will look
    # for tomorrow. Checking here turns a silent mismatch into a visible one.
    expected = predictions_path(today_str)
    if not expected.exists():
        print(f"\n!!! Predictions were not written to {expected.name}")
        print("    The grader will find nothing tomorrow. Check that the")
        print("    predictor derives its filename from config.MODEL_VERSION.")
        return 1
    print(f"\n    predictions written: {expected.name}  "
          f"({int(preds['bet'].sum())} bet(s) above the edge threshold)")

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
        # ROI through the tested odds math, not the decimal-on-American
        # arithmetic that produced -376% in the old rolling log.
        _step("Grade ROI to date", grade_all, required=False)
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
