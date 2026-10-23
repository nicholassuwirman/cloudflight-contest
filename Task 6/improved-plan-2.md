# Level 6 – Improvement Plan 2 (follow-up to `improved-plan-1.md`)

> **Audience:** the agent that will keep improving the solver in this folder.
> `plan.md` §2 (machine semantics) is still the ground truth and unchanged. `improved-plan-1.md` describes the
> engine; this document records what was **built and measured** while implementing it, which ideas turned out to be
> dead ends, and what to do next, ranked by expected lines saved per hour of work.
> **[MEASURED]** = observed by running code in this folder. **[HYPOTHESIS]** = design bet; confirm by measurement
> before investing heavily. Score = total lines, lower is better.

---

## 0. TL;DR

* Current verified totals **[MEASURED]** (all programs: exactly K acceptances, last line = K-th, checked by `simulate_ref`):

  | file | plan-1 start | now | first build T1 | repeat copies | #repeats | lines / repeat |
  |---|---|---|---|---|---|---|
  | small | 107 | 107 | 26 | 81 | 48 | 1.69 |
  | medium | 6033 | **6000** | 847 | 5153 | 1835 | 2.81 |
  | large | 5405 | **4936** (`best/`) / 4940 (out file) | 1650 | 3286 | 499 | 6.59 |
  | enormous | 5663 | **4572** | 3241 | 1331 | 70 | 19.0 |

  **Do first:** `out_level-6_3-large.txt` is stale (4940): a graph run was interrupted after improving `best/3-large.json`
  to 4936. Regenerate the out file from `best/` (step 1 of §9).
* **Where the remaining gain is** (from exact CP-SAT checks, §2):
  * **medium is close to optimal.** Every sampled first build (21/21) and every sampled steady-state cycle is proven
    optimal. Low priority.
  * **enormous first builds are the big pool (3241 lines),** then large first builds (1650) and the large repeat copies of
    mid-size patterns (where CP-SAT could not decide).
  * **Structural finding:** the beam almost never uses chord painting (a coincident pair). Only **2%** of enormous
    build ticks have one, and enormous builds take **1.81×** the number of chords on their best axis, versus 1.23× on large.
    Dense blobs are worst (enormous #1: 297 cells, 21 chords, T1 = 49). This is the main algorithmic opportunity (§5).
* **Ranked work items:**
  1. **Anytime improvement driver (§4)** combining three operators that each gave measured gains on top of the current
     best: extra symmetric portfolio passes (−1.6% / −3.2% enormous), suffix re-planning (−3.4% enormous, −1.1% large),
     copy-graph DP on K≥2 enormous (−2.2%). Cheap to build, since all the pieces exist. Expected: enormous −5…8%.
  2. **Make chord painting reachable (§5)**, via a lookahead / "paint potential" term in the beam score, macro-moves for
     painter setup, and CP-SAT sweep seeds. Highest ceiling (hundreds of lines on enormous), but [HYPOTHESIS].
  3. **CP-SAT as an optimality oracle and a small-instance solver (§6):** certify patterns as optimal so no CPU is
     spent on them; solve small first builds exactly; compute a valid per-copy lower bound.
  4. **Proper tuning bench (§7)**: the previous weight sweep was noise-limited.

---

## 1. What exists now (read before changing anything)

| file | content |
|---|---|
| `l6core.py`, `l6model.py`, `l6cycle.py` | unchanged (see plan-1 §1). |
| `l6beam.py` | main numba beam. **Bug fixed:** the stage-1 candidate buffer re-compacted on every child once a huge key tie at the threshold filled it (O(n²)). Now an overflow halves the threshold bucket uniformly at random and admits new tied children at rate 2^-rshift. **Effect [MEASURED]: enormous ~40× faster and better at equal W** (bench of 10 patterns at W=100: 430 → 394). New param `jit` (random tie-break in stage 2; measured useless, keep 0). `l6beam_orig.py` = pre-fix copy, for reference only. |
| `l6multi.py` | `beam_multi`: single-copy beam from an arbitrary config with an empty board; records up to `MAXE` accepting endpoints (deduplicated by canonical config), continues `slack` layers after the first acceptance. Same overflow fix. |
| `l6graph.py` | copy-graph DP (plan-1 §3): `CopyGraph` with canonical nodes (axis pairs unordered), swap-mask remapping of segment actions, `seed_program` (adds the incumbent's segments, so the DP is never worse than the incumbent), best-first expansion with priority `min_k f[k][X] + (K−k)·alpha·cbar`, exact K-edge DP with numpy. `solve_graph(case, incumbent, budget, W, ...)`. |
| `l6sym.py` | D6 symmetric restarts: `transform_case`, `sym_info` (emitter permutation derived programmatically from the corner/direction match; F/B swap for reflections), `map_back`; `solve_sym` runs `nsym` symmetries × `nport` portfolio entries, passing the incumbent length as `maxT`. All 12 maps verified with `simulate_ref` on every result. |
| `l6post.py` | big-int line-space replayer `Replayer.accepts(acts)` (tracks skirt matter too) + greedy line deletion / adjacent-line merge with truncation at the K-th acceptance. ~1% on enormous, 0 on medium/large. |
| `run.py` | `python run.py <tag> [W] [CAP] mode=beam|sym|graph|post k=v ...`; workers receive the incumbent `best[i]`; monotone merge into `best/`; rewrites and re-validates the out file **only when the whole run finishes**. |
| `exp_cpsat_cycle.py` | CP-SAT model of c ticks of the machine (one-hot positions, toggles via XOR, hits per relevant cell, exact-2 / ≥3, built-state recurrence with destruction, skirt hard or `soft`). Cyclic mode = steady-state copy cycle (config returns, axis swap allowed). Reused by the other experiments. |
| `exp_cpsat_t1.py` | "can the first build finish in ≤ c ticks from the start state?" |
| `exp_window.py` | fixed-end window repair with CP-SAT (dead end, see §3). |
| `exp_suffix.py` | suffix re-planning experiment (works, see §4.2). |
| `exp_sweep.py` | painter-pair sweep feasibility with CP-SAT (§5.3). |

Dependencies: numba 0.68, numpy 2.4, **ortools 9.15** (installed with pip; CP-SAT). 12 cores.

**Invariants (unchanged):** every program entering `best/` is verified with `simulate_ref`; out files are generated
only from `best/`; one blank line between programs plus a trailing newline; deterministic seeds.

**Pitfall found:** `multiprocessing.Pool` reuses workers. Never mutate module-level state (e.g. `l6sym.PORTFOLIO`) in a
worker; pass the parameters per job instead.

---

## 2. Exact facts established with CP-SAT [MEASURED]

All with the beam's hard skirt rule (skirt never receives matter) unless noted; 12 workers.

| question | instance | result |
|---|---|---|
| steady-state cycle, one copy per period, c=5 (incumbent 6) | medium #50 (K=19) | **INFEASIBLE** (53 s); also infeasible with the skirt relaxed (`soft`, 113 s) → 6/copy is optimal |
| cycle c=4 (incumbent alternates 4/5) | medium #119 | INFEASIBLE (13.5 s) → the 4/5 rhythm is optimal for single-copy cycles |
| cycle c=5 (incumbent 6) | large #71 | INFEASIBLE (15 s) |
| cycle c=6 / c=8 | large #122 / #59 | UNKNOWN after 240 / 300 s |
| first build T1−1 | 21 medium patterns (every 10th) | **21/21 INFEASIBLE** (12 s total) → medium first builds are optimal |
| first build T1−1 | large #67, #106, #71; enormous #42 | all INFEASIBLE (0.1–35 s) |
| first build T1−2 | enormous #56 (R=10, 53 cells) | UNKNOWN after 180 s |
| strict sweep: painter pair from one end, 1 chord / tick, horizon 31 (incumbent 49) | enormous #1 | INFEASIBLE both directions (16 s) |

Consequences:
* Medium: the steady-state copy cost and the first build are optimal on every sample tried. What is left is the
  transition into the cycle and patterns that have not been checked. Expect ≤1% from medium. Spend CPU elsewhere.
* CP-SAT decides small instances in seconds but stalls at roughly R≥8 with ≥40 cells, or horizons ≥ ~14. Use it as an oracle
  for small cases and inside restricted models, not as the main solver for big builds.
* The hard skirt rule did not cost anything on #50. A soft-contamination search (plan-1 §8) is therefore low priority.

---

## 3. Measured dead ends (don't repeat)

* **Wider beam beyond W≈1600** on enormous: W=1600 → 310, W=6400 → 313 (bench of 10). Width is saturated; the heuristic is
  the bottleneck.
* **Random tie-break in stage 2 (`jit=1`)**: mixed bench 1484 → 1491 (worse/noise).
* **Weight sweep (iB, iP, wS, wM) on a 10-pattern bench**: totals 312–340 with no consistent winner. The spread is noise.
  Use it as a portfolio, not as a tuning result (but see §7).
* **Copy-graph DP on medium**: 6033 → 6000 (4 patterns improved, 10 s each). From incumbent nodes the shortest copy is the
  same at W=300, 3000 and 20000. **On large**: 3 of the 32 largest patterns improved, −4 lines in total (30 s each).
* **Fixed-end window repair with CP-SAT** (state at a → exact config at b, built ⊇ original, w−1 ticks): 0 wins on
  enormous #77 (w = 6, 10, 14) and large #127 (w = 5); infeasibility is proven in 1–16 s. Sanity check: same-length windows are
  feasible, so the model is right. Pinning the end config exactly is too restrictive. Use **free-ended** re-planning instead (§4.2).
* Line deletion / merging (`l6post.py`): ~1% on enormous only. Keep it as the final pass, but don't expect more.

---

## 4. Anytime improvement driver (do first; all parts exist)

Write `improve.py` that runs indefinitely (or for a time budget) over patterns, applies operators to the incumbent, merges
improvements into `best/` immediately (verified, monotone) **and rewrites the out file periodically** (every N
improvements, plus at exit and on Ctrl-C), so interruptions never leave a stale out file.

### 4.1 Operators and measured value (each relative to the current best; samples of 12–20 patterns)

| operator | where | measured gain | cost |
|---|---|---|---|
| symmetric portfolio, extra parameter set (12 syms, W=1600) | enormous | PORTFOLIO[1]: 617 → 607 (−1.6%); PORTFOLIO[2]: 617 → 597 (−3.2%) | ~96 s wall / 20 patterns |
| suffix re-planning (cuts at 30/50/70%, W=3200, 2 param sets) | enormous | 617 → 596 (−3.4%) | 50 s wall / 20 patterns |
| suffix re-planning (cuts at 20/40/60/80%) | large | 357 → 353 (−1.1%) | 27 s wall / 19 patterns |
| copy-graph DP (W=800, slack 2, 90 s) | enormous K=2 | 413 → 404 (−2.2%) | 90 s per pattern |
| post_optimize | enormous | ~1% | <1 s |

The gains overlap only partly. Run them in rotation.

### 4.2 Suffix re-planning (generalise `exp_suffix.py`)
* Replay the incumbent to tick t (`state_at`: config, built ∩ P, acceptance count; assert the skirt is clean), run
  `beam_search` from that state with `maxT = len(best) − t`, splice the result if it is shorter, verify.
* Cut points: random t, biased towards the first build and towards copies whose length is above the pattern's median copy length.
  Parameter set and seed drawn from the portfolio. Iterate until there has been no improvement for M tries.
* **Under symmetry:** transform the case **and the prefix** (`map_back` inverse: permute the columns with the inverse
  permutation, swap F/B for reflections), re-plan the suffix in the transformed frame, map back. That gives 12× more
  diversity for the same operator.
* Variant: cut at t, re-plan with a **different K-target**. E.g. for K=2 enormous, re-plan from the state just after the first
  acceptance with the copy-graph (`beam_multi` endpoints) so that the second copy is chosen jointly.

### 4.3 Scheduling
* Value per pattern = (current lines − best known lower bound) × (1 if not certified optimal). Lower bounds from §6.
  Spend CPU proportionally. Mark patterns as **certified optimal** (CP-SAT proved T1−1 infeasible and every repeat copy is
  at a proven cycle optimum) and skip them.
* Long-tail fix: sort jobs by the previous run's measured time, longest first; `imap_unordered` with chunksize 1.

---

## 5. Chord painting: the main structural opportunity [HYPOTHESIS, strong evidence]

### 5.1 Evidence [MEASURED]
* Coincident-pair usage during first builds: **enormous 50 / 3241 ticks (2%)**, large 78 / 1650 (5%).
* T1 versus `minLines` (number of distinct chords of the pattern on its best axis): correlation **0.93** on enormous
  (T1~nP: 0.88). Σ T1 = 3241 against Σ minLines = 1786 → **1.81×**; large 1.23×.
* Sparse patterns already beat the chord count (crossings are efficient, ratio 0.64–0.9), so the loss is concentrated on
  **dense blobs**: e.g. enormous #1 (297 cells, 21 chords) T1 = 49 (2.33×), #40 (106 cells, 12 chords) 21 (1.75×),
  #78 (120 cells, 13 chords) 20 (1.54×). The 30 densest enormous patterns account for **1187** of the 3241 build lines.
* Skirt cells per chord on the best axis (each needs a 3rd beam to block it while a coincident pair paints that chord, ≤4
  blockers available): enormous chords with ≤4 skirt cells = 70%, with >4 = 30%. Only 14/140 enormous patterns (but 133/190
  large) have every chord at ≤4.

Why the beam misses it: painting a chord needs both sliders of an axis on the same line, active, **and** blockers on
every skirt cell of that chord in the same tick. Setting this up takes several ticks with no new built cells, so the
`built`-dominated score drops those states, and any premature activation is hard-pruned (skirt hit). This is a plateau /
delayed-reward problem, which is what lookahead and pilot methods address (Duin & Voß; Bertsekas rollout).

### 5.2 Approaches, cheapest first
1. **Paint-potential feature in the score.** For each stage-2 candidate compute `paintNext` = max over the ≤3 axes and the
   painter lines reachable in one move of (new P cells from a coincident paint of that chord, if its skirt cells can be
   blocked by the current/next positions of the other 4 sliders, else 0). Add `wPaint · paintNext` to the score.
   Cheap version: precompute per (axis, line) the chord's P-cell bitset and skirt-cell list once per pattern.
   Only evaluate on stage-2 candidates.
2. **1-ply lookahead on the top candidates** (plan-1 §6.5): for the best ~200 stage-2 candidates, compute the best safe
   child's gain (max new cells over that candidate's ≤4096 children, with the same safety test) and add it to the score. NW≤6 →
   ~25k word-ops per candidate; affordable at W≤800.
3. **Macro-moves.** Allow an extra successor type per state: "set up painter on axis a, line l" = the multi-tick
   sequence that brings both sliders to l and toggles them on in the tick where blockers sit on all skirt cells (blocker
   positions chosen greedily, nearest sliders first). Insert the macro's end state into layer t+len with its actions. This
   is the classic macro-operator remedy for plateaus.
4. **CP-SAT sweep seeds** (§5.3), injected as extra start states / incumbents for suffix re-planning.

Acceptance test: on the 30 densest enormous patterns, T1 must drop. Target Σ T1 1187 → ≤1000.
The coincident-pair tick ratio should rise. Keep the change only if the full-bench total does not get worse on sparse patterns
(make `wPaint` a portfolio dimension if needed).

### 5.3 CP-SAT sweep feasibility (partially measured)
* `exp_sweep.py`: fix both sliders of the best axis to a strict sweep (move to the first chord from the centre, toggle on,
  then exactly one chord per tick); CP-SAT chooses the other 4 sliders. **Enormous #1, horizon 31: INFEASIBLE in both
  directions (16 s).** The strict schedule is too rigid. This does not refute the idea.
* Next experiment (not run): only require that the painter pair is coincident at every tick (`L[a][t]==L[a+3][t]`,
  `A[a][t]==A[a+3][t]`), schedule free, horizon 38 / 44 (incumbent 49), 400 s limit. If feasible → extract the program,
  verify, and use it as a seed. If CP-SAT stalls, try a two-stage method: first choose the chord order and the per-tick blocker
  targets with a small DP, then let CP-SAT only fill slider moves.
* Also try painting the chords of **two axes** in alternation (a blob's boundary rows on one axis are often cheaper to block on another).

---

## 6. CP-SAT as oracle and small-instance solver

Build `l6exact.py` from `exp_cpsat_cycle.py` (keep the model function; add warm-start hints from the incumbent with
`model.AddHint`, which CP-SAT needs to find feasible solutions at the incumbent length).

1. **Certification pass** (all files, time limit 20–60 s per query): T1−1 from the start state. If INFEASIBLE, the first build is
   optimal. For each pattern with K>1, a cyclic `c*−1` check where c* = the incumbent's best steady copy length. Store the
   results in `best/<tag>.cert.json` and let the driver skip certified parts.
2. **Valid per-copy lower bound**: `cmin` = minimum c such that *some* free start config (non-cyclic, empty board) admits a
   one-copy segment of length c. Then `lines ≥ T1* + (K−1)·cmin`, where T1* is the exact or bounded first build. This gives a
   real `lines − LB` gap per pattern, which drives §4.3. (Cyclic infeasibility alone is **not** a valid bound for finite K,
   since transitions can be cheaper than the cycle mean.)
3. **Exact solving where it is fast**: whenever T1−1 or c−1 comes back FEASIBLE, the solution is a better segment. Extract the
   actions (the `U` variables as in `exp_window.py`), splice, verify. Most useful on large R=6–7 patterns with small K-copy
   costs where the beam may be off by one.
4. Free-ended final segment: "from the state at tick t, accept within r ticks" (no end config constraint). This is the
   CP-SAT analogue of suffix re-planning and avoids the over-constrained fixed-end window. Try r ≤ 8 on enormous builds.

Research basis: planning as satisfiability with step-indexed encodings and horizon search (Rintanen et al.), and LNS with
a CP repair step (Shaw 1998; ALNS, Ropke & Pisinger).

---

## 7. Tuning that actually measures something

* The 10-pattern bench has noise of ±10 lines (±3%) between parameter sets, so it cannot rank them. Use **paired comparisons
  against `best/`** on ≥40 patterns per file (every 3rd/4th) and ≥2 seeds. Report the sum of `new − best` and the count of wins/losses.
* Tune per file: enormous and large builds behave differently (crossing-heavy versus chord-heavy).
* Add the new score features (§5.2: `wPaint`, the lookahead weight) to the portfolio instead of fixing one value.
  Rafbill's AHC021 notes found several independent medium-width searches slightly better than one wide search. That matches the
  measured W saturation here.

---

## 8. Lower priority / optional

* **Speed** only matters once the heuristic improves: Euler-tour beam (avoid copying states; 2–5× reported) or `numba.prange`
  over parents. W is saturated today, so this buys more portfolio runs per hour, nothing more.
* **Diversity control** (cap states per canonical config, Diverse Beam Search groups): untested. Cheap to try as a portfolio entry.
* **Soft skirt contamination with erase**: CP-SAT showed no gain on #50. Only revisit if a §6 query shows that the soft
  model is feasible where the hard one is not.
* **Medium**: only the transition copies (first build → steady cycle) remain. Use the copy-graph DP with CP-SAT-generated edges
  (exact shortest copy from each incumbent node), or skip.

---

## 9. Work order and checkpoints

1. Regenerate `out_level-6_3-large.txt` from `best/` and validate (expect 4936, bad=0). Commit.
2. `improve.py` driver with operators sym-portfolio, suffix re-planning (incl. symmetry), graph DP for K≥2, post_optimize,
   and periodic out-file export. Run enormous then large for ≥1 h each. Checkpoint: enormous ≤ 4350, large ≤ 4850.
3. §6.1 certification pass (cheap) → skip certified patterns from then on.
4. §5.2 paint-potential feature and 1-ply lookahead → test on the 30 densest enormous patterns (paired). Checkpoint: Σ T1 on that
   subset drops ≥10%. If yes, re-run the whole driver with the new beam in the portfolio.
5. §5.3 relaxed CP-SAT sweep for the densest patterns; §6.3/6.4 exact repairs for small ones.
6. Medium last (§8).

After every step: append the per-file totals to a "Results log" section at the end of this file, and commit.

**Hard requirements, unchanged:** exactly K acceptances; the last line is the K-th acceptance; validated with `simulate_ref`;
output format as `out_level-6_0-example.txt`; deterministic seeds; monotone `best/`.

---

## Results log

| date | step | small | medium | large | enormous |
|---|---|---|---|---|---|
| 2026-10-23 | plan-1 baseline | 107 | 6033 | 5405 | 5663 |
| 2026-10-23 | overflow fix + graph DP (medium) + 12-sym portfolio W=1600 (large, enormous) + partial graph (large) | 107 | 6000 | 4936 | 4572 |

---

## Sources

* Beam search practice in contests: [Euler tour beam search, Rafbill (Codeforces)](https://codeforces.net/topic/123913/en1); [AHC021 editorial by Rafbill](https://atcoder.jp/contests/ahc021/editorial/6681); [Amplifying human performance in combinatorial competitive programming (arXiv 2411.19744)](https://arxiv.org/pdf/2411.19744)
* Lookahead / plateau remedies: [Pilot, Rollout and Monte Carlo Tree Search methods (arXiv 1210.0374)](https://ar5iv.labs.arxiv.org/html/1210.0374); [Duin & Voß, The Pilot Method (record)](https://core.ac.uk/works/17015849); [construction metaheuristics lecture notes, SDU](https://imada.sdu.dk/~marco/Teaching/AY2019-2020/DM865/assets/dm865-constr-meta-handout.pdf)
* Diversity: [Creative Beam Search / Diverse Beam Search discussion (arXiv 2405.00099)](https://arxiv.org/pdf/2405.00099)
* Dominance and duplicate pruning: [Revisiting Dominance Pruning in Decoupled Search, Gnad AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17403); [Finding a Dominating State in a Haystack, ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/42834)
* Planning as SAT / horizon search: [Rintanen, Heljanko, Niemelä, Planning as satisfiability: parallel plans and algorithms for plan search (AIJ 2006)](https://users.aalto.fi/rintanj1/papers/RintanenHN06.pdf); [Parallel Encodings of Classical Planning as Satisfiability](https://users.aalto.fi/~rintanj1/papers/RintanenHN04.pdf); [Rintanen, Planning as satisfiability: heuristics (AIJ 2012)](https://users.aalto.fi/rintanj1/papers/Rintanen12AIJ.pdf)
* CP-SAT: [OR-Tools CP-SAT code samples](https://nbviewer.org/github/google/or-tools/blob/6.7.x/examples/notebook/code_samples_sat.ipynb); [OR-Tools constraint optimisation docs](https://developers.google.com/optimization/cp)
* LNS with CP repair: [OR-Tools manual: job-shop LNS](https://lia.disi.unibo.it/Staff/MicheleLombardi/or-tools-doc/user_manual/manual/metaheuristics/jobshop_lns.html); [ALNS lecture notes (Ropke & Pisinger)](https://imada.sdu.dk/u/marco/DM841/slides/dm841-hr-alns.pdf)
* Plan-1 sources (iterative beam search, minimum mean cycle, window-based plan optimisation, NRPA, discrete tomography) remain relevant; see `improved-plan-1.md`.
