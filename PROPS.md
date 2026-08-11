# Player props

Branch `props/player-props`. Moneyline is closed — see [ROADMAP.md](ROADMAP.md), nine approaches and a decisive blend test showing the model adds 0.00004 nats to the market price. Props were the strongest remaining candidate on a structural argument: a book sets one moneyline per game with its best model and 200+ props a night with far less attention each.

## The data problem, and how it was solved

There is no widely-known free archive of historical prop lines, and the first four searches said so. Prop lines are 200+ per game night and change constantly, so nobody stores them the way game lines get stored.

They do exist. **[meherkalra/NBA-NFL-predictor](https://github.com/meherkalra/NBA-NFL-predictor)** carries FanDuel prop quotes with line value, both sides' prices, and timestamps:

| | |
|---|---|
| Coverage | 65 game days, 2023-03-20 → 2023-10-30 |
| Markets | points, rebounds, assists, threes, steals, blocks |
| Book | FanDuel |
| Rows | 304,032 total; 108,987 on points alone |
| Cost | free |

This makes The Odds API's paid historical props ($59/month) unnecessary for a first test.

## The trap in that data

Each player-game carries a median of **seven distinct line values** across timestamps. These are not alternate lines — the line is moving, and **the later quotes are live in-game**.

| quote selected | over-rate |
|---|---|
| last | 27.97% |
| **first (pre-tip)** | **46.86%** |

A 27.97% over-rate is not a bookmaker's line; it is a line that has already watched part of the game. 46.86% is what a real pre-game line looks like once the right-skew of scoring is accounted for. This single selection decision separates a garbage result from a valid one.

## Result: all six markets

Out-of-sample, calibration offset fit on the earliest third only, ROI from the actual posted odds rather than an assumed −110.

| market | archive rows | OOS n | over-rate | hit % | **ROI** |
|---|---|---|---|---|---|
| points | 108,987 | 1,365 | 45.1% | 51.50% | **−4.40%** |
| rebounds | 58,854 | 1,405 | 49.5% | 55.09% | **−0.34%** |
| assists | 45,988 | 1,351 | 49.3% | 55.96% | **+0.37%** |
| threes | 36,273 | 1,394 | 40.5% | 57.03% | **−7.42%** |
| steals | 29,005 | 1,429 | 48.1% | 56.61% | **−4.07%** |
| blocks | 24,925 | 1,436 | 35.2% | 67.27% | **−6.25%** |

**The volume hypothesis is not supported.** Blocks and steals are the least-traded markets and among the worst performers.

## Two findings worth keeping

**1. The 57% was an artifact.** Against a self-invented, minutes-aware line the points signal hit 57.4% over 68,798 player-games. Against FanDuel's actual line it hits 51.5%. We were beating our own rough estimate, not a bookmaker. Minutes mean-reversion is real — it is simply already priced, exactly as everything in the moneyline work turned out to be.

**2. Hit rate and ROI diverge sharply on props, and the divergence flatters the model.** Blocks hits **67.27%** and loses **6.25%**. A 35.2% over-rate means most blocks props are heavy under-favourites priced near −300: you win often and are paid nothing. Only ROI computed from real posted odds means anything here. Any prop result quoted as a hit rate should be treated as unreported until the ROI is shown.

## Assists, and why it is not a finding

The least-bad market, and it improves with edge size:

| threshold | bets | ROI | ±1 s.e. |
|---|---|---|---|
| all | 1,351 | +0.37% | 2.77 |
| ≥0.5 | 765 | +4.03% | 3.48 |
| ≥1.0 | 379 | +6.93% | 5.02 |
| ≥1.5 | 199 | +8.09% | 5.69 |

The best figure is **1.4σ**. By month: −2.07%, −1.21%, +6.63% — **1 of 3 profitable, with October carrying the total.** That is the signature of every false positive in this project.

## Scope

Stated plainly so this is not over-read in either direction. The archive is 65 game days and roughly 1,350 out-of-sample bets per market. **A small genuine edge could not be resolved at this sample size.** What can be said:

- The specific minutes-based projection tested is not an edge in any of the six markets
- Props are now testable properly and for free, and any new idea gets an answer in about ten minutes
- More data would be needed to resolve anything smaller than roughly ±3% ROI

## Reproducing

```bash
python -m src.props.projection          # live lines vs projection, logs predictions
```

Historical archive downloads to `data/props_raw/` (gitignored — 48 MB, and the
source repo declares no licence, so it is not redistributed here).
