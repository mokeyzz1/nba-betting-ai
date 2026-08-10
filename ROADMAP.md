# Roadmap

## The gap, measured on four recent seasons

Tested on 4,835 games from 2021-22 through 2024-25 against real closing prices, using the current model with injuries and recency weighting:

| edge threshold | bets | ROI | ±1 s.e. |
|---|---|---|---|
| 0% | 4,835 | −4.18% | 1.95 |
| 3% | 3,340 | −4.81% | 2.38 |
| 5% | 2,531 | −4.98% | 2.76 |
| 10% | 1,146 | −2.53% | 4.22 |

Head to head on the same games, the market wins on every measure: accuracy .6782 vs .6585, log loss .5998 vs .6188, AUC .7322 vs .7084.

**The deficit is not mostly vig.** Re-settling the identical bets at the de-vigged fair price — bookmaker margin removed entirely — still loses 0.76% at a 3% edge and 0.94% at 5%. When this model disagrees with the market, the model is usually the one that is wrong.

This finding overrides the earlier version of this document, which ranked cheaper execution as comparable in size to the whole deficit. It is not. A free bet at a fair price would still lose money.

## What has been ruled out

Recorded so nobody re-tests them:

- **Cheaper venue alone.** Prediction markets save roughly 3 points of vig. At zero vig the model still loses at every threshold below 10%.
- **Public betting percentages.** The recent odds data carries money-share and ticket-share per side. Following the sharp side (money% exceeding ticket%) returns −7.3% to −12.4%. Fading the public returns −5.8% to −6.3%. Correlation between the sharp signal and the market's own error is −0.028 on 4,426 games: the market has already priced it.
- **Bigger models.** XGBoost at depth 3 and 5 both scored worse than regularised logistic regression on the same features.

## What remains

Confidence matters more than size: a mechanical 2% beats a speculative 5%. But note that nothing below is currently known to close a 4-5 point gap.

---

## Ranked levers

### 1. Multi-book line shopping — *high confidence, +1 to 3%*

**Status:** module built ([`src/features/line_shopping.py`](src/features/line_shopping.py)), not yet wired into the live pipeline.

The old fetchers read `bookmakers[0]` — one arbitrary book. Measured on a live slate, the first book was not the best price on **60% of sides**, with a median gain of **+3.27%** per unit staked.

This is the only lever that pays regardless of model quality. A better price is worth the same whether the model is excellent or mediocre, and it requires no new research. Expect less in NBA than the measured WNBA slate — NBA markets are more liquid and books cluster tighter — so plan on 1–2%.

**Work:** route the daily pipeline through `line_shopping.fetch`, record which book offered each price, and grade the live log against the price actually available rather than a single book's number.

**Caveat:** capturing this requires accounts at several books, and books restrict accounts that consistently take the best number.

---

### 2. Player-level features — *partly done; lineup work remains*

**Injuries: shipped.** Official NBA injury reports, weighted by each player's pre-game minutes ([`src/backtest/injuries.py`](src/backtest/injuries.py), [`injury_features.py`](src/backtest/injury_features.py)). Over 6,106 games this improved accuracy .6592 → .6639 and AUC .7090 → .7155, better in every season where the feature is live. Real, consistent, and much smaller than the remaining gap.

**Lineups: not started.** [`pbpstats`](https://pypi.org/project/pbpstats/) (MIT) parses play-by-play into possessions with the five players on the floor. That is the one substantial source of information the model still cannot see: it treats a team as a fixed unit, so it cannot distinguish a starting five from a bench-heavy rotation on the second night of a back-to-back.

**Why moderate confidence:** the market prices injuries and rest too, and fast — the injury result above is evidence of exactly that, a real signal worth only a fraction of a point. The edge, if any, is in second-order effects on rotation and pace rather than in "star is out", which every book already knows.

**Work:** extend `dataset.py` with lineup-derived features under the same leakage discipline (strictly pre-tipoff, `.shift(1)` on every window), then re-run the harness.

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
| Today, measured | −4.2% to −4.8% |
| + line shopping (1–2%) | −3% |
| + prediction-market pricing (~3%) | −1% to −2% |

Stacking both known levers still does not reach break-even, because both act on execution cost and the deficit is mostly model error. They are worth taking once there is an edge to protect. They do not create one.

Two constraints still apply if that ever changes: edge at these magnitudes needs on the order of 1,000 wagers to detect, and books limit accounts that win.

## Sequence

**Phase 1 — Consolidate.** *Done.* 31 dead files removed, the version mismatch that stopped the pipeline ever grading itself is fixed, both broken ROI evaluators are replaced, mock data is opt-in, and the model refuses to predict on frozen features.

**Phase 2 — Line shopping.** *Built, not wired to a live season.* `predict_daily.py` prices at the best available number across books; it has not run against real games because the season starts in late October.

**Phase 3 — Model.** *Injuries done, lineups open.* See lever 2.

**Phase 4 — Live loop.** Daily automation for the October start, logging odds at several timestamps so closing-line value can be measured going forward. CLV is the reason to bother: it reads edge in roughly 50 bets where ROI needs on the order of 1,000.

**Known gap in the live path.** `predict_daily.py` currently hardcodes `rest_diff`, `b2b_diff` and the three injury features to zero, because the schedule join and same-day injury fetch are not wired. That is 5 of 15 features frozen at serve time — the same defect this project removed from the Hybrid Elite model, and it must be fixed before the pipeline runs for real.

## The honest question

Four seasons of testing say this model loses to the closing line by about 4 points, and by about 1 point even with the vig removed. Nothing on the list above is known to close that.

That is the normal outcome. NBA moneyline aggregates enormous amounts of money and information, and independent models built on public data usually land here. Two responses are reasonable, and neither is giving up:

- **Change the target.** Less efficient markets — props, lower-liquidity lines — are where inefficiency actually lives. That needs prop odds data this project does not have.
- **Keep it as an instrument.** The harness, the leakage checks and the honest reporting are the valuable artefacts. They make this a credible engineering project regardless of whether it ever turns a profit.

What is no longer reasonable is expecting execution improvements to close a model-sized gap.

---

## Principles

Adopted after the audit, and the reason these numbers can be trusted more than the last set:

1. **Never synthesise a price or an outcome.** If real data is missing, the pipeline fails loudly rather than filling the gap.
2. **Every feature must be knowable before tipoff.** Enforced automatically in `dataset.py`.
3. **One implementation of odds math.** All of it in `src/backtest/odds.py`, covered by assertions.
4. **Report the median and the standard error.** A mean can be carried by one outlier; a return without an error bar is not a result.
5. **Prefer the measurement over the model.** A worse model that is honestly measured is more valuable than a better one that is not.
