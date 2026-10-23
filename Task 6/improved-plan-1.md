# Level 6 – Improvement Plan 1 (follow-up to `plan.md`)

> **Audience:** the agent that will improve the existing solver in this folder.
> `plan.md` is still the reference for the problem semantics (§2 there is verified and unchanged). This document
> covers what has been **built and measured** since, and what to do next, ranked by expected gain per hour of work.
> **[MEASURED]** = observed by running the current code. **[HYPOTHESIS]** = design bet you must confirm by measurement.

---

## 0. TL;DR

* Current submitted scores **[MEASURED]**: medium **6033**, large **5405**, enormous **5663** (small 107, example 23,
  versus 446 for the reference example). All are verified by the slow reference simulator, so every program accepts exactly K times
  and ends on the K-th acceptance.
* Where the lines are **[MEASURED]**:

  | file | total | first build T1 | repeat copies | #repeat copies | lines / repeat copy |
  |---|---|---|---|---|---|
  | small | 107 | 26 | 81 | 48 | 1.69 |
  | medium | 6033 | 847 | **5186** | 1835 | 2.83 |
  | large | 5405 | 1782 | **3623** | 499 | **7.26** |
  | enormous | 5663 | **3965** | 1698 | 70 | 24.3 |

* The three biggest levers, in order:
  1. **Copy-graph DP (new, exact decomposition, §3).** After an acceptance, the relevant board is *empty* (all pattern cells
     removed and the skirt is always clean), so the entire state is just the slider configuration. The K-copy problem is
     therefore exactly a **shortest K-edge path in a graph of "post-acceptance configurations"**. The current beam
     chooses each copy's end configuration greedily (myopically). Replace it with: generate many alternative copy endpoints
     per node, then run an exact DP over K. Targets: medium repeats (5186) and large repeats (3623).
  2. **Spend the idle compute (§4).** Enormous ran at W=100 in **195 s wall** on 12 cores, and large at W=200 in 149 s.
     Use anytime iterative widening per pattern with a time budget, plus **12 free symmetric restarts** (D6 rotations
     and reflections of the board, §5). Targets: enormous T1 (3965) and large T1 (1782).
  3. **Beam-selection fixes (§6).** Biased tie handling at the stage-1 threshold, no diversity control, and untuned
     weights. These are cheap to fix and tune with the existing `tune.py`.
* Things already **ruled out by measurement**. Don't spend time on them:
  * Wider beam on small: W=300, 1000 and 5000 all give **107** exactly.
  * Period-1 and period-2 steady-state cycles on medium: an exhaustive enumeration (`l6cycle.py`) shows the beam already
    finds every one of them (total possible gain **1 line**).
  * Repeating a loop found inside an existing beam path (same acceptance config seen twice): total gain **9 lines** on medium.

---

## 1. Current code (read these first)

| file | content |
|---|---|
| `l6core.py` | input parser, `Geometry` (direct port of the sandbox `rebuildPerimeter` / `isInGrid` / ray cells), **`simulate_ref`** (slow, faithful sandbox port; ground truth), `read_output` / `write_output`, `validate_file`. |
| `l6model.py` | `LineModel(R)`: emitter offset o∈[-R,R] → normalized line index ℓ∈[0,2R] (0 and 2R are dead/silent), `sign[e]` (F means ℓ+sign). Asserts that every in-window ray equals the full chord. `PatternModel(case)`: compact bitsets over relevant cells (pattern P first, then skirt S): `lineBits[3][2R+1][NW]`, `Pm`, `Sm`, `cellLine[nP][3]`. NW≤1 for small/medium, ≤3 for large, ≤6 for enormous. |
| `l6beam.py` | numba kernel `beam_search(...)`. Layered beam over ticks; per parent, enumerates per-axis distinct canonical next states (`axis_options`, ≤16 per axis, ≤4096 joint children). **Hard prune**: E2∩S≠∅. Stage 1: integer key `acc*KB + iB*built + iP*pot` (pot = missing P cells under ≥1 active beam), and an incremental histogram threshold keeps ≥CAP children. Stage 2: up to `2*CAP` best keys are re-simulated, deduplicated with a canonical hash (the two sliders of an axis are unordered), and scored `key − wS*psum − wM*pmax` (reach-distance penalty, `penalty()`); top W kept. Returns at the **first** layer where any child reaches K acceptances. |
| `l6cycle.py` | exhaustive period-1/period-2 cycle enumerator (per-axis tables deduplicated by bitsets, then a triple join). Diagnostic only. |
| `l6solve.py` | `solve(case, **params)` → beam → `actions_to_program` → asserts with `simulate_ref`. `DEFAULT = W=300, CAP=3000, iB=4, iP=1, wS=0.05, wM=0.5`. |
| `run.py` | `python run.py <tag> [W] [CAP] [k=v ...] [only=1,2,3]`: multiprocessing over patterns (largest first), keeps the **best verified program per pattern** in `best/<tag>.json` (monotone, never replaced by a longer one), writes `out_level-6_<tag>.txt`, re-validates the whole file. |
| `tune.py` | `python tune.py '<json list of param dicts>' [mixed|medium|large|enormous]`: runs a fixed subset (every 7th medium pattern and every 12th large; every 14th enormous) and prints the line total per parameter set. Does not touch `best/`. |

Runtime facts **[MEASURED]** (12 cores, numba 0.68, Python 3.13): medium W=300 → 42 s; large W=200 → 149 s (slowest
pattern 114 s); enormous W=100 → 195 s (slowest 195 s). Small W=5000 → 156 s (so cost is ~linear in W).

**Invariants to keep:** every candidate program is checked with `simulate_ref` before it enters `best/`; the output is
regenerated only from `best/`; one blank line between programs plus a trailing newline (as `write_output` does now).

---

## 2. Lower bounds and gap (so you know when to stop)

* Per repeat copy, `c ≥ 1`, and `c ≥ 3` for every pattern where `l6cycle.find_cycles` finds no period-≤2 cycle.
  **[MEASURED]** on medium: the repeat-cost lower bound is **4456** (now 5186). **93 patterns need c≥3**: they cost 3324 now
  against a bound of 2565. All remaining medium gain is in those 93 patterns, at most ~730 lines (less in reality).
* Large and enormous have no useful bound yet. Cheap ones to add and log per pattern:
  `T1 ≥ max_p reach(p from start)` (the same reach formula as `penalty()`, evaluated at the start state), and
  `T1 ≥ ⌈|P| / maxCellsPerTick⌉` where maxCellsPerTick = max over all single-tick stamps of any config
  (compute by enumerating per-axis line multisets, as `l6cycle` does, for one phase).
* Log `lines − LB` per pattern after every run. Then spend CPU on the patterns with the largest gap × K.

---

## 3. Copy-graph DP (highest expected gain for medium and large)

### 3.1 Why this is exact
After an acceptance tick, every pattern cell is `x` (→ `O` next tick) and the skirt holds no matter (hard-prune
invariant). Matter outside P∪S never matters. So the post-acceptance state is **fully determined by the 6-slider config**
(ℓ_e, a_e), using exact emitter identities for replay. Define the graph G:

* node = post-acceptance config X (plus the special start node: all ℓ=R, all inactive, empty board);
* edge X→Y with weight w = number of ticks of a segment that starts in (X, empty), contains **exactly one** acceptance,
  on its last tick, and leaves config Y.

The optimal program is the minimum-weight path of exactly K edges starting at the start node. Segments concatenate
exactly, because the state at each junction is identical. For large K this is a minimum-mean-cycle problem (Karp / Howard
policy iteration, see sources). With K ≤ 19 here, a plain DP over k is exact and simpler:
`f[k][Y] = min_X f[k-1][X] + w(X,Y)`.

The current solver is a greedy approximation of this. Its beam keeps the states that finish copy k earliest, and among
those it cannot tell which end configuration makes copy k+1 cheap. That shows in medium paths like gaps `[3,4,3,4,…]` and
`[5,6,6,5,6,…]` **[MEASURED]**.

### 3.2 Edge generation: multi-endpoint single-copy beam
Add a mode to `beam_search` (new numba function, or a flag):
* start from an arbitrary config (L0, A0) with an empty board (already supported via the `L0, A0, built0, acc0` arguments);
  target K=1;
* **do not return at the first acceptance.** Record every accepting child (deduplicate by exact config) together with its depth and the
  parent chain. Keep the accepting children out of the beam (they are leaves). Continue for `slack` more layers (start with 2), so slightly
  slower copies that end in better configurations are also captured;
* return up to `M` endpoints (M ≈ 32–128), each as (depth, end config, action list). If an edge has more endpoints than M, prefer distinct
  "shapes" of the end config (different per-axis line multisets), not just the shallowest ones.

Path reconstruction for many endpoints: the existing `parArr/actArr` back-pointers already allow it. Record (layer, beam
index of parent, action) for each accepting child.

### 3.3 Graph growth and DP
```
nodes = {start}; frontier = [start]
while budget left and frontier:
    X = pop the frontier node with the best f-value (f = best known cost to reach X with k copies, any k)
    for (d, Y, actions) in multi_endpoint_beam(X): add edge X→Y; push Y if new
run the DP over K on the graph; reconstruct; concatenate segments; verify with simulate_ref; keep if shorter.
```
* Canonicalize node keys (sorted pair per axis, as in `canon_hash`). Two configs that differ only by swapping the two emitters of
  an axis are equivalent, but the segment's actions must then be remapped: swap the u-codes of emitters e and e+3. The
  actions are in ℓ-space, so a swap only exchanges which emitter does what. `actions_to_program` handles the F/B signs.
  The simplest correct version: store exact configs, and use canonical keys only for merging, with a stored permutation.
* Also normalize inactive sliders: an inactive slider's ℓ still matters (the time to reach a line later), so keep it in
  the key.
* Budget for medium (R≤5, NW=1): a single-copy beam at W=300 takes milliseconds to tenths of a second, so 200–2000 nodes per pattern
  are affordable. Large: fewer nodes (50–200), since K≤6 and each beam costs seconds.
* Sanity check **[HYPOTHESIS]**: on the 93 medium patterns with c≥3 this should move several patterns from an average gap of 3.5–6
  toward the c=3 bound. Expected total gain is 150–500 lines on medium, and a few hundred on large (499 repeat copies at
  7.26 each).
* Enormous (K≤2): the DP is `min_X d(start→X) + d(X→any)`. Generate endpoints of the first build with `slack` and pick the one
  whose second copy is cheapest. This is cheap to add once 3.2 exists.

### 3.4 Optional: exact period-3 check (only for knowing when to stop)
Extend `l6cycle.py` to period 3. Per slider there are 10 closed action walks of length 3 (stay×3; the 6 orderings of +1,−1,stay;
and the 3 orderings of T,T,stay). Build per-axis tables deduplicated by the 3-phase bitsets, then join with pruning: after
choosing H and D, cells that are hit by ≥3 beams in all three phases can never be created, so prune; and cells with 0 hits from
H∪D in all phases must be covered by an E coincident pair. If c=3 is proven achievable for a pattern and the DP
reaches it, stop spending CPU on that pattern.

---

## 4. Spend the idle compute: anytime iterative widening

Research basis: iterative/anytime beam search (restart with geometrically growing width until the time limit) is
state of the art on several combinatorial problems; see Zhang's CABS, the iterative beam search used on the permutation
flowshop, and Lemons & Ruml's Rectangle Search.

* Per pattern: run W = 100, 200, 400, 800, … (CAP = 10·W) until a **per-pattern time budget** is used up. Keep the best
  result. Pass the incumbent length as `maxT` so hopeless runs stop early (the beam cannot return a longer result than `maxT`
  anyway).
* Budget by value: weight ∝ (current lines − LB). Enormous T1 is 3965 lines for 140 patterns. A full enormous pass at
  W=1600 is roughly 16× the W=100 cost, so ~50 min wall. Do that first.
* Maze-like patterns (e.g. enormous #64: 184 cells, 150 lines for K=2, solved in **2.3 s** at W=100 **[MEASURED]**) are
  cheap because the hard prune kills most children. Give them **much** larger widths (10k–50k), since the cost per layer is small.
* Parallelism: the run currently ends with a long tail (one 195 s pattern while 11 cores idle). Either sort jobs by
  measured time from the previous run, or add `numba.prange` over parents inside `beam_search` for single
  large patterns. Stage 1 then needs per-thread histograms and candidate buffers that are merged afterwards.

---

## 5. Free diversity: D6 symmetric restarts

The board, the emitter start corners (emitter e at perimeter index e·R) and the firing directions are invariant under
the 6 rotations of the hexagon. Rotating by 60° maps emitter e → e+1 (mod 6) and keeps F/B (clockwise stays clockwise).
The 6 reflections additionally swap F↔B and permute emitters. So for each of the 12 symmetries g:
`solve(g(pattern))` → map the program back with g⁻¹ (permute the 6 columns, swap F/B for reflections) → a valid program of
the same length for the original pattern. The beam's tie-breaking (parent order, histogram, random jitter) is not
symmetric, so these are 12 genuinely different restarts at no modelling cost.

* Cube coordinates for text cell (r,c), with dr=r−R and dc=c−2R: z=dr, x=(dc−dr)/2, y=−x−z. Rotation by 60°:
  (x,y,z)→(−z,−x,−y). Back: dr=z, dc=2x+z. Derive the emitter permutation **programmatically**: rotate each emitter's
  start cell and firing direction and match them. Then **verify** by simulating the mapped-back program with
  `simulate_ref` on the original pattern (it must give exactly K acceptances). Do not trust the formula alone.
* Use it together with the seed: e.g. budget = 12 symmetries × 2 seeds at the final width.

---

## 6. Beam-selection fixes (cheap; tune with `tune.py`)

1. **Threshold-bucket bias [HYPOTHESIS: important early in each copy].** In stage 1, every child whose key equals the
   threshold T is kept in *parent order*, so when keys tie massively (right after an acceptance, or early in a build where
   built=0 for everyone) the CAP slots are filled by the children of the first few parents. Fix: reservoir-sample within the
   threshold bucket (keep a uniformly random subset), or cap children per parent at ~CAP/W·4.
2. **Diversity control.** Cap the number of states per canonical config (e.g. ≤4), or per (config, built-count bucket).
   This is the beam-search analogue of Diverse Beam Search (Vijayakumar et al.): partition the beam into groups and
   penalize similarity to already-selected groups. In this problem, "similar" = same slider config or Hamming-close `built`.
3. **Score shape.** Stage 2 currently = `iB*built + iP*pot − wS*psum − wM*pmax`. Try an explicit *remaining-ticks estimate*
   `h = max(pmax, ⌈missing/ρ⌉)` with ρ = |P|/(observed ticks per copy) learned from the first run, ranking by `h` and
   then `psum`. Sweep `iB/iP ∈ {2,4,8}`, `wS ∈ {0, .02, .05, .1, .2}`, `wM ∈ {0, .25, .5, 1, 2}` on the `mixed` and
   `enormous` benches (one sweep ≈ minutes). Parameters may differ per file; store the winners in `DEFAULT` per tag.
4. **Better "potential" for maze patterns.** `pot` counts missing cells under ≥1 beam, but in mazes most such cells cannot
   be completed safely. Variant: count missing cells c where some single-slider move would make c an exact-2 cell
   **without** creating an E2 on a skirt cell. Evaluate it only for stage-2 candidates.
5. **2-ply tie-break.** For the top few hundred stage-2 candidates, compute the best one-tick gain (max new cells over that
   child's own safe children) and add it to the score. Cost ≈ 4096 child evaluations per candidate, so only use it with NW≤2
   or a small candidate set.

---

## 7. Post-optimisation: windowed re-planning (LNS)

Research basis: plan-improvement by local re-planning of windows (Nakhost & Müller's Plan Neighbourhood Graph Search, and
windowed re-planning such as R-WIN/CH-WIN, ICAPS 2016). Take an existing plan, re-optimize a sub-window, and repeat until
time runs out.

* Replay the best program and record the state (config, built, acc) after every tick. Pick a window [a, b], e.g. random,
  or around copies whose gap is above the pattern's minimum. Run a goal-directed beam from state a whose goal is
  **config == config_b, acc == acc_b and built ⊇ built_b** (a superset is fine). Accept the result if it uses fewer than b−a ticks; splice it in and re-verify the whole
  program with `simulate_ref` (a superset could cause an earlier acceptance. That is only better, but it must be re-checked
  and the program truncated at the K-th acceptance). The goal-directed beam needs the distance-to-target-config term
  `Σ_e |ℓ_e − ℓ*_e| + [a_e ≠ a*_e]` in its score.
* Cheap action elimination first: try deleting each line, and merging line t into line t+1 when no emitter uses both. Keep any
  change that still gives exactly K acceptances with the last one on the last line. Use the fast replay plus `simulate_ref` at the end.
* Expected gain **[HYPOTHESIS]**: small (1–3%) on medium (copies are already short), larger on enormous builds of 30–75 ticks.

---

## 8. Optional / experimental (only after §3–§6)

* **Soft contamination with erase** for maze patterns (enormous, large). Today any E2 on a skirt cell is pruned. Allow it
  with a penalty λ per dirty skirt cell (`dirty` bitset in the state; acceptance requires dirty=∅; a ≥3-hit on a dirty
  cell cleans it). First measure the fraction of children pruned per layer on #64-type patterns. If it is above 95%, this is
  worth trying. Otherwise skip.
* **Structured seeds for dense blobs** (plan.md §5.5/M5; discrete-tomography flavour: three line directions on a
  hexagonal/triangular lattice, cf. the Nagy et al. papers). A coincident pair paints a whole chord, and ≤4 other beams block
  run ends and holes. Generate a row/diagonal sweep schedule and inject its intermediate states into the beam as extra
  start states. Relevant only to patterns like medium #197 (51 cells, 8–10 ticks per copy) and the dense large/enormous blobs.
* **Policy-learning search** (NRPA / Beam-NRPA, Cazenave et al.): learn a per-(layer, action-feature) preference from the best
  sequences and bias the beam's tie-breaks with it. Only consider this if §3–§7 plateau.
* **Front-to-front bidirectional search** (Holte et al. MM; Kuroiwa & Fukunaga) for single copies, searching backwards from
  configs whose stamp completes P. Low priority: the backward state space (board ⊆ P) is large, and the copy-graph DP
  already handles the end-configuration problem.

---

## 9. Work order, checkpoints, acceptance criteria

1. **§6.1 + §6.3 sweep** (≤1 h). Checkpoint: the `tune.py mixed` total improves; adopt the parameters.
2. **§5 symmetry restarts** (≤1 h incl. verification) and **§4 iterative widening with a budget**. Run enormous and large overnight-style.
   Checkpoint: enormous < 5663, large < 5405.
3. **§3 copy-graph DP** (main engineering item). Implement the multi-endpoint beam, then the graph plus DP. Checkpoint on the 93 medium
   c≥3 patterns first: the repeat total falls from 3324 toward 2565.
4. **§7 LNS** on everything, time-boxed.
5. **§8** only if time remains.

After every step: `python run.py <tag> …` merges into `best/` monotonically, rewrites the out file and re-validates it.
Record per-file totals in a small table in this file (append a "Results log" section) so regressions are obvious.

**Hard requirements, unchanged:** exactly K acceptances; the last line is the K-th acceptance; validated with `simulate_ref`;
output format as `out_level-6_0-example.txt`; deterministic seeds.

---

## Sources (research used for this plan)

* Iterative / anytime beam search: [Iterative beam search algorithms for the permutation flowshop (arXiv 2009.05800)](https://arxiv.org/pdf/2009.05800); [Complete Anytime Beam Search applied to decision trees (arXiv 2508.06064)](https://arxiv.org/pdf/2508.06064); [Beam-stack search, Zhou & Hansen, ICAPS 2005](https://cdn.aaai.org/ICAPS/2005/ICAPS05-010.pdf); [Rectangle Search, Lemons & Ruml](https://liner.com/review/rectangle-search-anytime-beam-search-extended-version); [Triangle search, Ruml group, HSDIP 2023](https://www.cs.unh.edu/~ruml/papers/triangle-hsdip-23.pdf); [Beam Search: Faster and Monotonic (arXiv 2204.02929)](https://arxiv.org/pdf/2204.02929)
* Diversity: [Diverse Beam Search, Vijayakumar et al. (arXiv 1610.02424)](https://ar5iv.labs.arxiv.org/html/1610.02424); [AAAI-18 version](https://vision.soic.indiana.edu/papers/diversebeam2018aaai.pdf); [Determinantal Beam Search (arXiv 2106.07400)](https://arxiv.org/pdf/2106.07400)
* Minimum mean cycle (steady-state copy cost): [Computing Optimal Cycle Mean in Parallel on CUDA (arXiv 1111.0627)](https://arxiv.org/pdf/1111.0627); [Approximating the minimum cycle mean](https://eprints.cs.univie.ac.at/4098/1/mincyclemean.pdf); [Boost Howard cycle ratio](https://live.boost.org/doc/libs/1_81_0/boost/graph/howard_cycle_ratio.hpp); [LEMON graph library docs](https://lemon.cs.elte.hu/hg/lemon/rev/0a42883c8221)
* Plan improvement / LNS: [Window-based plan optimisation, HSDIP/ICAPS 2016](https://icaps16.icaps-conference.org/proceedings/hsdip16.pdf); [Continuing plan quality optimisation (JAIR, PNGS + BDPO2)](https://jair.org/index.php/jair/article/download/10966/26110/20454); [Very large-scale neighbourhood search (course notes)](https://homes.di.unimi.it/cordone/courses/2024-ae/Lez15-Materiali/VeryLargeScaleNeighborhoodSearch.pdf)
* Policy-learning search: [Beam Nested Rollout Policy Adaptation, Cazenave & Teytaud](https://www.lamsade.dauphine.fr/~cazenave/papers/CazenaveBeam.pdf); [NRPA, Rosin, IJCAI 2011](https://ijcai.org/Abstract/11/115)
* Bidirectional search: [MM, Holte et al., AAAI 2016](https://www2.cs.uh.edu/~ceick/ai/AAAI-2016-Best.pdf); [Front-to-Front Heuristic Search for Satisficing Planning, IJCAI 2020](https://www.ijcai.org/proceedings/2020/567)
* Discrete tomography on hexagonal/triangular grids (three line directions): [Binary Tomography on Triangular Grid Involving Hexagonal Grid Approach](https://i-rep.emu.edu.tr/items/7b7e9cc1-9303-4142-8ccd-482c2f30cfbb); [Discrete Tomography on the Triangular Grid based on Ryser's Results](https://i-rep.emu.edu.tr/items/b06df21b-13a3-4a31-afbc-1de131495263/full)
* Contest context: [Cloudflight Coding Contest 2026](https://internshala.com/competitions/cloudflight-coding-contest-2026/)
