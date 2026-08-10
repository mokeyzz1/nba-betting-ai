"""
Official NBA injury reports, collected and cached.

Fills the hole left by src/features/get_injury_data.py, which has always been
a stub returning {}. Data comes from the injury reports teams are required to
file with the league, via the `nbainjuries` package (MIT).

TIMING MATTERS AND IS EASY TO GET WRONG. The league issues reports throughout
the day and the late ones are the most accurate -- but a 5:30pm report is
published AFTER a 2:00pm tipoff, so training on it would leak the very
information the model is supposed to predict without. We therefore take the
08:00 ET report, which precedes every NBA tipoff (the earliest are around
noon ET, including games played abroad). It is slightly staler than an
evening report, and that is the correct trade: it is also what a bettor
placing a morning wager would actually know.

Coverage starts with the 2021-22 season; nothing earlier exists in this form.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
CACHE = BASE / "data" / "injuries"
CACHE.mkdir(parents=True, exist_ok=True)

# Tried in order. All are before any tipoff; later times are excluded on purpose.
REPORT_TIMES = [(8, 0), (4, 45), (5, 30), (8, 15)]

STATUS_WEIGHT = {
    "out": 1.00,
    "doubtful": 0.75,
    "questionable": 0.50,
    "probable": 0.25,
    "available": 0.00,
}


def fetch_day(day: datetime) -> pd.DataFrame | None:
    """One day's report as a DataFrame, or None if no report exists."""
    from nbainjuries import injury

    for h, m in REPORT_TIMES:
        ts = day.replace(hour=h, minute=m, second=0, microsecond=0)
        try:
            if not injury.check_reportvalid(ts):
                continue
            df = injury.get_reportdata(ts, return_df=True)
            if df is None or len(df) == 0:
                continue
            df["report_time"] = f"{h:02d}:{m:02d}"
            df["report_date"] = day.date().isoformat()
            return df
        except Exception:
            continue
    return None


def _cache_path(day: datetime) -> Path:
    return CACHE / f"injuries_{day.date().isoformat()}.csv"


def backfill(start: str, end: str, pause: float = 0.3, verbose: bool = True) -> int:
    """Cache reports across a date range. Resumable -- cached days are skipped."""
    d0 = datetime.fromisoformat(start)
    d1 = datetime.fromisoformat(end)
    day = d0
    got = missing = cached = 0

    while day <= d1:
        p = _cache_path(day)
        if p.exists():
            cached += 1
            day += timedelta(days=1)
            continue
        # The league publishes nothing in the deep offseason; skip Jul-Sep.
        if day.month in (7, 8, 9):
            day += timedelta(days=1)
            continue

        df = fetch_day(day)
        if df is not None:
            df.to_csv(p, index=False)
            got += 1
        else:
            missing += 1
            p.write_text("")          # negative cache: do not re-probe
        time.sleep(pause)

        if verbose and (got + missing) % 50 == 0 and (got + missing):
            print(f"    {day.date()}  fetched={got} empty={missing}")
        day += timedelta(days=1)

    if verbose:
        print(f"  done: {got} reports fetched, {missing} days with none, {cached} already cached")
    return got


def load_cached() -> pd.DataFrame:
    """Every cached report, concatenated and normalised."""
    frames = []
    for p in sorted(CACHE.glob("injuries_*.csv")):
        if p.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["status"] = df["current_status"].astype(str).str.strip().str.lower()
    df["weight"] = df["status"].map(STATUS_WEIGHT).fillna(0.0)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce", format="mixed")

    # "Lastname, Firstname" -> "Firstname Lastname", to match box-score naming.
    name = df["player_name"].astype(str).str.strip()
    swapped = name.str.split(",", n=1).apply(
        lambda p: f"{p[1].strip()} {p[0].strip()}" if len(p) == 2 else p[0]
    )
    df["player"] = swapped.str.replace(r"\s+", " ", regex=True).str.strip()

    # Two-way and G League call-ups are roster paperwork, not injuries. They
    # are left in: weighting by recent minutes zeroes them out naturally,
    # which is more robust than pattern-matching the reason text.
    return df


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2021-10-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-06-30"
    print(f"Backfilling injury reports {start} -> {end}")
    backfill(start, end)
    df = load_cached()
    print(f"\nrows: {len(df):,}")
    if len(df):
        print(f"dates: {df['report_date'].min()} -> {df['report_date'].max()}")
        print(f"\nstatus mix:\n{df['status'].value_counts().head(8)}")
