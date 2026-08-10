# Roadmap

## The gap, stated plainly

| | |
|---|---|
| Current position vs closing line | **−1% to −2% ROI** |
| Bookmaker margin being paid | 3.81% |
| Model accuracy vs market accuracy | 66.5% vs 68.0% |

The model is not far off. It is 1.5 accuracy points behind the market and paying a 3.8% toll. Closing that gap needs roughly **2–3 percentage points of ROI** from somewhere.

This document ranks the available sources by expected value and by how confident we can be that they exist. Confidence matters more than size: a speculative 5% is worth less than a mechanical 2%.

---

## Ranked levers

### 1. Multi-book line shopping — *high confidence, +1 to 3%*

**Status:** module built ([`src/features/line_shopping.py`](src/features/line_shopping.py)), not yet wired into the live pipeline.

The old fetchers read `bookmakers[0]` — one arbitrary book. Measured on a live slate, the first book was not the best price on **60% of sides**, with a median gain of **+3.27%** per unit staked.

This is the only lever that pays regardless of model quality. A better price is worth the same whether the model is excellent or mediocre, and it requires no new research. Expect less in NBA than the measured WNBA slate — NBA markets are more liquid and books cluster tighter — so plan on 1–2%.

**Work:** route the daily pipeline through `line_shopping.fetch`, record which book offered each price, and grade the live log against the price actually available rather than a single book's number.

**Caveat:** capturing this requires accounts at several books, and books restrict accounts that consistently take the best number.

---

### 2. Player-level features — *moderate confidence, unknown size*

**Status:** not started. `data/PlayerStatistics.csv` is 305 MB of real game logs and the model has never touched it.

The current 12 features are all team-level. Everything about *who is actually playing* is invisible to the model — and that is precisely where public markets are slowest to adjust.

Concretely available without any new data source:

- **Minutes trends as an injury proxy.** A rotation player whose minutes collapse over three games is hurt, resting, or benched. No injury API needed.
- **Lineup strength.** Aggregate recent player-level production for the players actually available, rather than treating the team as a fixed unit.
- **Player-level Elo or plus-minus**, rolled up to the matchup.
- **Star availability**, which moves lines several points and is the single largest source of price movement.

**Why moderate confidence:** the market prices injuries too, and fast. The edge is in *degree* — books adjust for a star being out, less reliably for the second-order effects on rotation and pace. Worth testing; not guaranteed.

**Work:** extend `dataset.py` with player-derived features under the same leakage discipline (strictly pre-tipoff, `.shift(1)` on every window), then re-run the harness. The harness already exists, so this is measurable the day it is built.

---

### 3. Bet opening lines, not closing lines — *moderate confidence, +1 to 2%*

**Status:** not started. Requires forward data collection.

Everything in this repo is graded against **closing** prices, which are the sharpest number a market produces — all information, all money, fully incorporated. Being 1.5 points behind the closing line is a much better result than being 1.5 points behind an opening line.

Opening numbers are softer, posted with less information and lower limits. A model that loses to the close may still beat the open.

**Work:** log odds at multiple timestamps daily from October, then grade the same predictions against open, mid-day, and close. This costs nothing but time and answers whether the edge exists earlier in the cycle. It also produces the closing-line-value metric that professionals use to judge a model before the money settles.

---

### 4. Softer markets — *speculative, potentially large*

Player props and less-liquid markets are meaningfully less efficient than NBA moneyline, which is among the most efficient markets in sport. That is where inefficiency actually lives.

**Blocker:** prop odds data. The historical endpoint is paid, and the prop models in this repo were all built against fabricated prices, so none of their results carry information.

**Do not start here.** Props are only worth attempting once the moneyline loop is honest end-to-end, because props multiply the number of ways to fool yourself.

---

### 5. Staking discipline — *does not create edge, protects bankroll*

Kelly sizing is already implemented in `src/prediction/predict_enhanced.py`. It converts an edge into growth and, more importantly, prevents overbetting when the model is confidently wrong.

This is why the calibration fix mattered. Kelly staked on inflated probabilities overbets systematically. Calibrated probabilities make the sizing honest even when the edge is thin.

---

## Realistic arithmetic

| | ROI |
|---|---|
| Today | −1.5% |
| + line shopping (conservative) | +0.5% |
| + any real gain from player features | +1% to +2% |

That path leads to **thin but positive**, not to the +15.3% this project once claimed.

Two things to be clear-eyed about at that level:

1. **Detection takes hundreds of bets.** At +2% ROI with typical variance, distinguishing real edge from noise takes on the order of 1,000 wagers. The 121-bet live sample says almost nothing on its own.
2. **Books limit winners.** A consistently profitable account gets stake-limited. This constrains the ceiling regardless of model quality.

---

## Sequence

**Phase 1 — Consolidate (before October)**
Remove the ~26 dead files (broken imports, old-laptop paths, duplicates, one empty file). Fix the daily pipeline: it currently saves predictions under one name and grades under another, so grading never runs, and a bare `except` hides the failure. Replace the two broken ROI evaluators with `src/backtest/odds.py`. Delete or quarantine every module that generates synthetic data in a live path.

**Phase 2 — Wire in line shopping**
Route live odds through the multi-book fetcher. Record the book behind every price. This is the highest certainty-per-hour work available.

**Phase 3 — Player features**
Build them against the existing harness. Measure honestly; keep what survives.

**Phase 4 — Restore the live loop**
Daily automation for the October season start, logging odds at multiple timestamps so closing-line value can be measured going forward.

---

## Principles

Adopted after the audit, and the reason these numbers can be trusted more than the last set:

1. **Never synthesise a price or an outcome.** If real data is missing, the pipeline fails loudly rather than filling the gap.
2. **Every feature must be knowable before tipoff.** Enforced automatically in `dataset.py`.
3. **One implementation of odds math.** All of it in `src/backtest/odds.py`, covered by assertions.
4. **Report the median and the standard error.** A mean can be carried by one outlier; a return without an error bar is not a result.
5. **Prefer the measurement over the model.** A worse model that is honestly measured is more valuable than a better one that is not.
