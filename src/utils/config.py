from pathlib import Path

# Base directory = root of your project (adjust as needed)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Paths to your folders
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
PERFORMANCE_DIR = BASE_DIR / "performance"

# Make sure folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model version: ONE definition, used for both writing and grading.
#
# This exists because the two drifted apart. run_pipeline.py graded
# "v4_3_enhanced" while predict_today_enhanced.py wrote "hybrid_elite", so the
# grader looked for a file that was never written, found nothing, and a bare
# `except` reported success anyway. The pipeline ran for months reporting
# "Pipeline completed successfully" while measuring nothing.
#
# Anything that writes a prediction file and anything that grades one must
# derive its name from here.
# ---------------------------------------------------------------------------
MODEL_VERSION = "hybrid_elite"


def predictions_path(date_str: str, model_version: str = MODEL_VERSION) -> Path:
    """Canonical prediction filename. Use this instead of formatting by hand."""
    return PREDICTIONS_DIR / f"predictions_{date_str}_{model_version}.csv"


def accuracy_path(date_str: str, model_version: str = MODEL_VERSION) -> Path:
    """Canonical grading filename."""
    return PERFORMANCE_DIR / f"accuracy_{date_str}_{model_version}.csv"
