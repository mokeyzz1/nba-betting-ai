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
