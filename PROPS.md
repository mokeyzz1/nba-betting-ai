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


## Prop line movement

The strongest signal found anywhere in this project was game-line movement (22σ). Props carry timestamps too — a median of 10 quotes per player-game — so the same question applies.

**The live-quote trap catches this too, and harder.** Using all quotes, movement correlates **+0.81** with the outcome. That is the scoreboard, not a forecast: the line rose because the player was scoring. Restricting to pre-tip quotes:

| cutoff | n | corr(move, points − opening line) |
|---|---|---|
| before 23:00 UTC | 1,402 | +0.4860 |
| 2h+ before | 1,157 | +0.3931 |
| 3h+ before | 1,068 | +0.2778 |

Still declining as the cutoff moves earlier, so some leakage persists even at the 23:00 UTC reference. The 3h+ figure is genuinely pre-game and plausibly real: a line collapsing from 20.5 to 8.5 means the player was downgraded, and that predicts his output.

**But the move being informative is not the same as it being tradeable** — once the line has moved, the old number is gone. The exploitable question is whether the market misprices *after* moving:

| | correlation |
|---|---|
| move vs points − **opening** line | +0.4116 |
| move vs points − **final** line | **−0.1915** (se 0.0319, 6σ) |

**Negative means the market over-reacts.** After a prop line moves, outcomes tend back the other way relative to the new number. Following the move loses at every threshold (−1.65% to −13.59%), consistent with overshoot.

Fading it looks better and still fails:

| threshold | bets | ROI | ±1 s.e. |
|---|---|---|---|
| 2+ pts | 509 | +0.28% | 5.38 |
| 3+ pts | 388 | +4.45% | 6.32 |
| 5+ pts | 219 | +12.92% | 7.90 |

By month at the 2+ threshold: **+42.42%, −7.55%, −1.86% — 1 of 3, with the smallest month (n=89) carrying the total.** The headline figure is 1.6σ.

**Correction — the first version of this section mixed books.** The archive carries seven books (barstool, betmgm, betrivers, caesars, draftkings, fanduel, pinnacle) and 60% of player-games are quoted by two or more. Grouping by `(date, player)` without `book_key` measured the spread *between* books as if it were movement over time.

Redone per book, pre-tip only, correlation of the move against `points − final line`:

| book | n | correlation | σ |
|---|---|---|---|
| **FanDuel** | 574 | −0.2397 | **5.7** |
| **DraftKings** | 556 | −0.1858 | **4.4** |
| Barstool | 292 | −0.1164 | 2.0 |
| BetMGM | 153 | −0.0870 | 1.1 |

**The over-reaction survives.** It is not a cross-book artifact — it appears independently in both major books at 4–6σ.

Fading it, FanDuel and DraftKings combined:

| threshold | bets | hit % | ROI | ±1 s.e. |
|---|---|---|---|---|
| 1+ pts | 985 | 49.54% | −1.22% | 4.00 |
| 2+ pts | 736 | 51.22% | +4.39% | 4.94 |
| 3+ pts | 582 | 52.75% | +8.44% | 5.73 |
| 5+ pts | 332 | 56.93% | +15.16% | 7.12 |

And the month breakdown at 5+, which is where the discipline bites:

| month | bets | ROI |
|---|---|---|
| 2023-03 | 38 | +51.78% |
| **2023-04** | **249** | **+2.25%** |
| 2023-05 | 18 | −1.16% |
| 2023-10 | 27 | +93.64% |

**April holds 75% of the sample and returns +2.25%, which is noise.** The headline is carried by 38- and 27-bet months; +93.64% on 27 bets is roughly five lucky wins.

## Status

**Established:** prop lines over-react within a single book. −0.24 at FanDuel (5.7σ), −0.19 at DraftKings (4.4σ), consistent across both, not a book-spread artifact.

**Not established:** that fading it is profitable. The largest reliable subsample says +2.25%.

Note what a 5-point move means here: the median line is 18.5, so it is a **27% move** — a player downgraded to questionable or on a minutes limit. Only 332 such cases exist in 65 game days. This is a rare-event strategy, and rare events need long samples.

**This is the one open question worth paying for.** The Odds API sells three NBA seasons of props from May 2023 for $59 — roughly 10× this sample, enough to separate +2.25% from +15%. Unlike every other paid option considered in this project, the question is now specific enough to justify it.


## Does fading a substantial move produce positive CLV?

The sharper version of the profit question. Closing line value reads an edge in far fewer bets than ROI, so if the over-reaction is real the line should *retrace* after a substantial move.

Method: within each `(date, player, book)` sequence of pre-tip quotes, find the first point where the line has moved ≥2 points from its open, fade it there, and measure where the line finishes. Sign convention matters — an UNDER bet is better with a *higher* line, an OVER bet with a *lower* one.

**Aggregate looks encouraging and scales with move size:**

| move size | n | mean CLV | % positive |
|---|---|---|---|
| 2–3 pts | 344 | +0.253 | 42.7% |
| 3–5 pts | 296 | +0.463 | 55.1% |
| 5+ pts | 101 | +0.911 | 66.3% |

Overall **+0.426 pts, 2.3σ**, positive in both books.

**It does not survive two checks.**

*Direction.* A genuine over-reaction should be symmetric. It is not:

| we fade | n | mean CLV | σ |
|---|---|---|---|
| line moved DOWN → bet OVER | 393 | **+0.875** | 3.2 |
| line moved UP → bet UNDER | 348 | **−0.080** | −0.3 |

The effect exists only when backing a player whose line collapsed. That is plausibly a real asymmetry — downgrade news causes an overshoot that upgrades have no mirror for — but it halves the strategy and doubles the burden of proof.

*Period.* The dominant month reverses it:

| month | n | mean CLV | σ |
|---|---|---|---|
| 2023-03 | 82 | +1.585 | 2.4 |
| **2023-04** | **456** | **−0.206** | **−0.8** |
| 2023-05 | 34 | +0.853 | 0.9 |
| 2023-10 | 169 | +1.485 | 5.8 |

April holds 62% of the sample and is negative. The aggregate is carried by March and October.

**Answer: no.** Fading substantial prop moves does not produce reliable positive CLV on this data. The over-reaction correlation remains real (−0.24 FanDuel, 5.7σ) but it does not convert into a better number at the close, and it does not convert into ROI.

**What would change the answer:** three seasons instead of 65 days. 741 qualifying events, only 393 in the direction that works, split across four months, cannot separate a real 0.9-point CLV edge from two lucky months. This remains the one question in the project where more data is worth buying.
