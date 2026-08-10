# NBA Betting Model

A machine-learning system for NBA game prediction, with an evaluation harness built to be hard to fool.

The model predicts game winners at **66.5% accuracy** on walk-forward validation across 13,831 games. Measured against real closing prices, it currently runs **1–2% behind the market** — it has genuine predictive signal but does not yet beat the line after vig.

Those numbers are lower than what this README used to claim. The reason is documented below, and it is the most useful thing in this repository.

---

## Honest performance

Walk-forward validation: for each test season the model is trained only on seasons that finished before it. Graded against real closing moneyline prices from 2008–2023.

| | model | market |
|---|---|---|
| Accuracy | 0.665 | 0.680 |
| Log loss | 0.6110 | 0.5924 |
| AUC | 0.732 | — |
| Expected calibration error | 0.026 | — |

**Betting simulation vs real closing odds** (13,831 games):

| edge threshold | bets | win rate | ROI | ±1 s.e. |
|---|---|---|---|---|
| 0% | 13,831 | 44.2% | −3.20% | 1.22 |
| 3% | 9,591 | 42.9% | −2.04% | 1.49 |
| 5% | 7,166 | 42.1% | −1.25% | 1.74 |
| 10% | 3,133 | 39.8% | −0.31% | 2.66 |

Reference points: flat-betting every home team returns −3.61%, every favourite −4.47%, and the average bookmaker margin is 3.81%.

**The model is not profitable against closing lines.** It is close to break-even at high conviction, and the gap to the market is smaller than the vig — but a losing system is a losing system, and this README will say so until that changes.

### Live results, March–April 2025

The pipeline ran unattended on a cloud VM for a month, posting picks before tipoff and grading them the next day.

| | |
|---|---|
| Graded bets | 121 |
| Accuracy | 76.0% |
| ROI (flat 1 unit) | +2.1% |

That is a real out-of-sample record, but 121 bets is a small sample: one standard error on the ROI is roughly ±9%, so +2.1% is not statistically distinguishable from zero. High accuracy with low return means it was mostly backing favourites — right often, paid little.

---

## What the audit found

This project was rebuilt after an audit of all 88 Python files. Six defects materially changed what the numbers meant. They are documented because the failure modes are common and mostly invisible.

**1. Backtests graded the model against odds it invented.** The "market" price was the model's own probability plus Gaussian noise, and the bet trigger was `edge > 0.10` — where edge *was* that noise. It fired precisely on games the model already had right. This produced the previously advertised +61.4% and +59.7% returns.

**2. Two systems predicted a target they fabricated.** With real scores missing, `actual_total` was computed from pace and offensive rating — the same features fed to the model — then noise added and over/under labels cut from it. The model was learning to invert a formula written four lines earlier.

**3. Live ROI tracking used decimal-odds math on American odds.** A winning bet at −150 was scored as `−150 − 1 = −151` units. This is why the old rolling ROI log reports values like −376%. The model looked like it was losing catastrophically; it wasn't.

**4. Train/serve skew on the implied-probability feature.** Training computed `1 / home_odds` (for −150 that is −0.0067); serving used the correct conversion (0.60). Three of fourteen features arrived at prediction time ~100× off the scale the model learned, effectively disabling them in production.

**5. Look-ahead leakage from season-aggregate stats.** Team advanced stats were joined on `(team, season)`, attaching end-of-season ratings to games played in November — so each game's own result was baked into its own features. Compounding it, every training script used a shuffled random split on time-ordered data.

**6. Silent mock data in live paths.** Feature getters substituted randomly generated stats during the offseason without failing, and one returned unseeded `np.random` values on every call in season too. A prediction file from August 2025 pairs real sportsbook odds with fabricated team ratings.

The current pipeline is built so that these cannot recur silently: the dataset builder fails the build on constant features or any feature correlating above 0.6 with the target, and all odds math routes through a single tested module.

---

## How it works

```
data/  ──►  dataset.py  ──►  harness.py  ──►  honest ROI
             │                  │
             │                  └── walk-forward: train on prior seasons only
             └── pre-tipoff features only, leakage-checked
```

**Features** are all knowable before tipoff. Sequential Elo (updated after each game, snapshotted before), rolling form over 5- and 10-game windows with every window shifted by one game, season-to-date record, rest days, and back-to-back flags — expressed as home-minus-away differentials.

**Training** is recency-weighted with a four-year half-life. Without it the model inherits a ~58% home-court edge from decades of history and over-predicts the home side against today's ~55%, which cut expected calibration error by 58% when corrected.

**Odds** come from all available bookmakers, not the first one returned. Best-price selection across ~7 books was worth a median +3.27% per unit staked on a measured live slate — comparable in size to the model's entire deficit to the market.

**The model deliberately does not use odds as a feature.** A model fed the price mostly learns to restate it, and any apparent edge becomes noise around the market. Excluding it makes the comparison meaningful.

---

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp        # macOS: xgboost needs OpenMP
```

Create `.env` with an [Odds API](https://the-odds-api.com) key:

```
ODDS_API_KEY=your_key_here
```

Run the backtest:

```bash
.venv/bin/python -m src.backtest.harness
```

Backfill recent seasons from the NBA API:

```bash
.venv/bin/python -m src.backtest.backfill
```

---

## Layout

| path | purpose |
|---|---|
| `src/backtest/` | Dataset builder, walk-forward harness, odds math, calibration |
| `src/features/` | Live feature and odds fetching, incl. multi-book line shopping |
| `src/prediction/` | Daily prediction entry points |
| `src/training/` | Historical model training scripts |
| `src/evaluate/`, `src/monitor/` | Post-game grading and rolling performance |
| `data/` | Historical games, odds, and player statistics |
| `nba_dashboard.py` | Streamlit dashboard |

Data note: 71,291 games (1946–2026), of which 17,280 carry real closing odds (2008–2023). Historical odds for 2024-25 and 2025-26 are not available on The Odds API free tier, so those seasons can validate accuracy and calibration but not ROI.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the plan to close the gap to the market.

Near-term:

- Multi-book line shopping wired into the live pipeline
- Player-level features from the unused 305 MB of game logs — minutes trends as an injury proxy, lineup strength
- Consolidation: ~26 files are dead (broken imports, stale paths, duplicates) and are slated for removal
- Restore the daily automated loop for the October season start

---

## Caveats

This is a personal research project, not investment advice and not a product. Sports betting markets are close to efficient; NBA moneyline is among the most efficient of them. The realistic ceiling for a well-built independent model is a thin edge that takes hundreds of bets to distinguish from zero. Any system claiming otherwise — including earlier versions of this one — is measuring itself wrong.

**Tech:** Python · scikit-learn · XGBoost · pandas · nba_api · The Odds API · Streamlit
