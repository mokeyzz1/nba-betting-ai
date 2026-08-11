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

**The decisive one first.** Every other test asked whether the model *beats* the market. The sharper question is whether it adds anything *to* the market, since a weak but independent signal is monetisable even when it loses head to head. Blending the model into the de-vigged market price, walk-forward over 17,709 priced games:

| weight on model | blend log loss | vs market alone |
|---|---|---|
| 0.00 (market only) | 0.59155 | — |
| **0.05** | **0.59151** | **−0.00004** |
| 0.10 | 0.59156 | +0.00001 |
| 0.20 | 0.59196 | +0.00041 |
| 0.50 | 0.59537 | +0.00382 |

The optimal weight is 0.05 and the gain is 0.00004 nats. Against a 4.25% vig an exploitable contribution needs roughly 25–50× that. This should have been the first test run, not the last: it says directly that the model holds no information the price lacks, which is what the .898 correlation, the +0.011 residual and the failed decorrelation were each saying indirectly.

**The −0.82% ROI at a 5% edge threshold is therefore not a near miss that better features close.** It is the return from betting noise at better-than-average prices. NBA moneyline outcome prediction is finished.


- **Cheaper venue alone.** Prediction markets save roughly 3 points of vig. At zero vig the model still loses at every threshold below 10%.
- **Public betting percentages.** The recent odds data carries money-share and ticket-share per side. Following the sharp side (money% exceeding ticket%) returns −7.3% to −12.4%. Fading the public returns −5.8% to −6.3%. Correlation between the sharp signal and the market's own error is −0.028 on 4,426 games: the market has already priced it.
- **Bigger models.** XGBoost at depth 3 and 5 both scored worse than regularised logistic regression on the same features.
- **Decorrelating the model from the bookmaker.** Hubáček, Šourek & Železný, *Exploiting sports-betting market using machine learning* (Int. J. Forecasting, 2019) reports cumulative NBA profit from three ingredients: a decorrelation term in the training loss, a CNN aggregating player-level statistics, and portfolio-theoretic bet sizing. Their claim is that "an accurate model is unprofitable as long as it is correlated with the bookmaker's predictions".

  Implemented by residualising every feature against the de-vigged market probability, sweeping the strength from 0 to 1 over 20,007 priced games. It degrades monotonically:

  | λ | AUC | corr w/ market | ROI @5% |
  |---|---|---|---|
  | 0.0 | .7156 | .898 | −0.82% |
  | 0.4 | .6868 | .771 | −2.08% |
  | 1.0 | .5409 | .160 | −3.35% |

  The reason is the finding. Our model correlates .898 with the market; remove that component and AUC falls to .541, barely above chance. Decorrelation presupposes the model holds independent information, and ours does not. Same conclusion the residual test reached from the other direction.

- **Player-level information as an independent signal.** The remaining Hubáček ingredient, and the cheap version of it was tested before committing to a CNN. Built roster-*shape* features from 53,539 team-dates — star concentration (top-1 and top-3 production share), depth, production inequality, rotation size — quantities a team power rating cannot express. The decisive question was whether they reduce correlation with the market.

  | | AUC | corr w/ market | residual signal |
  |---|---|---|---|
  | team-level only | .7156 | **.898** | +.0088 (1.2σ) |
  | + roster shape | .7158 | **.899** | +.0132 (1.8σ) |

  Market correlation moved by .001. The book already prices roster shape. A CNN might find interactions these summary statistics miss, so this is evidence rather than proof — but it does not justify days of work, and their result predates a decade of bookmaker modelling (their data is 2007-2014).

- **Optimising for line-movement prediction.** Line movement has two components: information (news, injuries, sharp money correcting a bad number), which predicts outcomes, and flow (the public piling onto popular teams), which does not. Flow is the *more predictable* of the two, so a flexible learner chases it. Gradient boosting reaches movement correlation +0.334 against ridge's +0.234, while its outcome correlation collapses from +0.045 to +0.010. Movement prediction is therefore not a valid proxy target, and the chain "movement corr x 0.187 = outcome corr" holds only for the informative component. Optimise outcome correlation directly.

- **An exploitable bias in the opening spread.** The correlation between the opening spread and the cover margin reads +0.106 (11σ), which looks like favourites being systematically mispriced. It is an artifact: `cover = margin + spread`, so the two share a term. The money test settles it — always backing the underdog against the opening spread returns **−6.48%** over 10,603 bets at a 48.99% win rate against the 52.38% needed, losing in **0 of 11 seasons**. At the closing spread, −4.33%.

- **Switching among major US sports.** Same open/close archive covers NFL, NHL and MLB. On a common footing (probability space, open → close):

  | sport | games | closing vig | move→outcome | market log loss | learned open→close |
  |---|---|---|---|---|---|
  | NHL | 13,665 | 3.28% | +0.0639 | 0.6711 | +0.0022 |
  | MLB | 25,569 | **2.82%** | +0.0387 | 0.6809 | +0.0009 |
  | NBA | 12,023 | 3.80% | **+0.0861** | **0.5979** | +0.0037 |

  MLB is a point cheaper but its market log loss is 0.681 against a 0.693 coin flip — those sports are close to unforecastable for everyone, bookmaker included. NBA is the only one where outcomes are meaningfully predictable, and it is the most expensive. No major US sport is the soft target.

  This also corrects an earlier claim in this document. "Beat the open, not the close" assumed meaningful room between them. The market improves its log loss by 0.0009 to 0.0037 from open to close — **opening lines are already ~99% as sharp as closing lines.** The room is far smaller than implied.

- **Predicting the market's error directly.** Following Hubáček & Šír, *Beating the market with a bad predictive model* (arXiv:2010.12508), profit does not require beating the market's accuracy — it requires model errors that are *decorrelated* from the market's. Tested by training a ridge regression on our full feature set against the target `home_win - market_fair_prob`, walk-forward over 4,835 games. Correlation between predicted and actual market error: **-0.0057**, against a standard error of 0.0144. Betting the largest predicted mispricings returns -4.76% to -7.15%.

  This is the most informative negative result in the project. Our features are not decorrelated from the market — they *are* the market's information. Elo, form, rest and injuries are all priced. It follows that no better model of the same inputs can profit here: not more features of this kind, not a stronger learner, not better calibration. That avenue is closed, and closing it is worth more than another inconclusive experiment.

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

### 3. Bet opening lines, not closing lines — *tested, mostly closed*

**Status:** tested. The premise was largely wrong.

The reasoning was that closing prices are the sharpest number a market makes, so a model losing to the close might still beat the open. Free opening lines (SportsbookReview archive, 12,023 NBA games) let this be measured rather than assumed.

Two results, pulling opposite ways.

**The premise is weaker than assumed.** The market improves its own log loss by only 0.0037 between open and close in NBA — and 0.0009 to 0.0022 in MLB and NHL. Opening lines are already about 99% as sharp as closing lines. There is far less room there than "bet the open" implies.

**But the model does have closing line value.** Correlation between its disagreement with the opening line and the market's subsequent move toward that side is **+0.0792 (8.2σ, n=10,800)**. At a 5% disagreement the line moves the model's way 54.7% of the time, averaging +0.171 points. That is real and it is the only thing in this project that kept measuring significant.

It is still not enough. Betting the opening spread at −110 returns −4.21% (50.2% win rate against the 52.38% needed); the closing spread returns −4.70%. Opening beats closing, consistently with the CLV result, and both lose.

**Best configuration found:** movement model minus its flow component, outcome correlation **+0.0549** against the **~0.115** needed. About 48%.

**What remains here:** CLV is the right metric to track going forward, because it reads an edge in roughly 50 bets where ROI needs on the order of 1,000. Developing it further requires live data — timestamped odds at several points per day — which historical archives cannot supply.

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

**Live path: fixed.** `predict_daily.py` previously hardcoded `rest_diff`, `b2b_diff` and the three injury features to zero — 5 of 15 frozen at serve time, the same defect removed from the Hybrid Elite model. Rest and back-to-backs now derive from each team's last game date in `dataset.current_team_state()`; injuries come from `injury_features.live_burden()`, which fetches the day's official report and applies the same minutes-weighted definition as the backtest. Verified on a mid-season slate: 0 of 15 features constant. A guard now raises if more than a third of features are constant across a slate, so this cannot recur silently.

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
