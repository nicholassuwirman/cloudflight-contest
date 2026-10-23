# Level 6 – "Machine Time Isn't Cheap" – Implementation Plan

> Audience: the agent that will write the code. **No solution code exists yet.** Everything below is research:
> what the problem really is, which parts were *verified* against the provided files/sandbox, and the recommended
> (most-optimised, yet realistic-runtime) way to solve it. Items tagged **[VERIFIED]** were checked by running code
> against `out_level-6_0-example.txt` / `sandbox-level-6.html`. Items tagged **[HYPOTHESIS]** are design bets that
> the implementer must confirm by measurement (see §11).

---

## 0. TL;DR

* Level 6 is an **optimisation level**: for every pattern output a program (list of 6-token lines, one token per
  emitter) that makes the machine **accept the pattern exactly K times**. **Score = total number of lines over all
  programs (lower = better).** Output is one file per input file.
* The provided example solution is a very naive "single crossing-point walks cell by cell" program: 116 + 56 + 274 =
  **446 lines** for the 3 example patterns (and it even wastes 1–9 trailing lines after the last acceptance).
  Big improvements are available.
* The machine can be reduced to a clean model: **6 sliders, each choosing one of 2R-1 full chords ("lines") of the hexagon,
  each with an on/off flag. A cell becomes matter when exactly 2 beams cross it, is destroyed when ≥3 do.**
* Recommended approach (details in §6–§9):
  1. A bit-exact fast simulator + validator (ported from the sandbox JS) – the foundation.
  2. A guaranteed-valid **baseline** (single "pointer" = crossing of two beams walks through the pattern).
  3. The main engine: **beam search over ticks in 6-slider configuration space**, with hash-dedup, a
     reach-distance heuristic and contamination ("skirt") pruning. Multi-copy mode for K>1.
  4. A **cycle finder** for K>1 (steady-state loop / ping-pong), because for the *medium* file (ΣK = 2045 copies) the
     cost of *one extra line per copy* is worth ~2000 lines.
  5. Post-optimiser (tail trim, tick deletion + re-verify), then a final independent verification pass over every program.
* Where the points are (see §3 table): medium = K-dominated (optimise the *per-copy cycle*), large/enormous =
  build-dominated (optimise the *first build*), small = trivial but also worth exhaustive search.

---

## 1. Problem recap (from `level-6.pdf`)

* Input: `N`, then `N` blocks: header `R rows cols K` (radius, number of rows = 2R+1, number of columns = 4R+1,
  required count), then `rows` lines of pattern text (chars `-`, `*`, `O`, `M`), blocks separated by one empty line.
* The pattern is the **same size as the board**; it is compared **positionally** (no shifting). All `M` cells are
  connected (the sandbox validates this).
* Output: `N` programs, each = list of instruction lines; each line = 6 tokens separated by single spaces, token ∈
  `{F, B, T, -}` for emitters 1..6 in order. **One empty line between programs** (the example output file also ends
  with an empty line after the last program – replicate the format of `out_level-6_0-example.txt` exactly).
* Each program starts from the empty board, emitters in their corners (1 left-middle, 2 top-left, 3 top-right,
  4 right-middle, 5 bottom-right, 6 bottom-left), all inactive. Program must produce **K acceptances**.
* Score = total number of instruction lines (all lines count, including all-`-` lines). The sandbox additionally
  shows "Instructions"/"Cycles" counting only non-no-op tokens/lines, but the PDF states the score is the number of
  lines, so **every line costs 1 regardless of content** – a no-op line is as expensive as a busy one.
  ⇒ *Pack as many simultaneous actions as possible into each line.*
* The sandbox `sandbox-level-6.html` (open in a browser, paste input + solution) is the reference visualiser.
  Its JS is the ground truth for semantics (§2). `Simulation accepts X of K` is shown if too few; the sandbox does not
  complain about more than K, but **target exactly K and end the program on the line of the K-th acceptance**
  (safest for an unknown judge).

### Input files / statistics **[VERIFIED by script]**

| file | N | radius R | K min / max / Σ | pattern cells (M) min / avg / max | distinct patterns |
|---|---|---|---|---|---|
| 0-example | 3 | 3,3,4 | 1 / 4 / 8 | 5 / 9.7 / 16 | 3 |
| 1-small | 10 | 3 | 1 / 9 / 58 | 2 / 6.0 / 14 | 10 |
| 2-medium | 210 | 3–5 (75/73/62) | 1 / 19 / **2045** | 4 / 12.2 / 59 | 207 |
| 3-large | 190 | 6–8 (R=6: 65, R=7: 62, R=8: 63) | 1 / 6 / 689 | 6 / 30.6 / 131 | 190 |
| 4-enormous | 140 | 9–12 (40/35/34/31) | 1 / 2 / 210 | 18 / 90.4 / **297** | 140 |

Patterns are dense, connected blobs, often with isolated `O` holes inside (large/enormous), sometimes touching the
board's inner ring. Only 3 patterns in `medium` repeat (207/210 distinct) → caching by pattern is nearly useless,
but cheap to add.

---

## 2. Exact machine semantics **[VERIFIED]**

Derived from `sandbox-level-6.html` (`applyEmitterStep`, `cleanUpGrid`, `detectAndAcceptPatterns`, `testPattern`,
`acceptAndDeletePattern`, `rebuildPerimeter`) and cross-checked by re-implementing it in Python and replaying the
provided example output: accepted at ticks `[29,69,109]` (K=3), `[55]` (K=1), `[55,125,195,265]` (K=4) – exactly K
each, so my reading of the semantics is right.

### 2.1 One tick (= one instruction line)
1. Clean-up of the previous tick: cells that were *accepted* (`x`) or *destroyed* (`d`) become `O`; freshly created
   (`m`) become `M`. (Effect: acceptance/destruction is visible for exactly one tick, but the next tick's beams already
   see an empty cell → **a pattern cell that was just accepted can be re-created on the very next tick.**)
2. All 6 instructions execute simultaneously: `F` position+1 (clockwise), `B` position-1, `T` flip active flag,
   `-` nothing. One instruction per emitter per tick ⇒ **an emitter cannot move and toggle in the same tick.**
3. Beams: every *active* emitter fires a straight ray from its (border) cell in its fixed direction
   (1→E, 2→SE, 3→SW, 4→W, 5→NW, 6→NE). `hit[cell] += 1` for every interior cell on the ray (border / padding
   cells are ignored, the emitter's own cell is not part of the ray).
4. For every hit cell that currently is `O` or `M`:
   * `hit == 2` → `O` becomes matter (`m`), `M` stays `M`;
   * `hit >= 3` → the cell becomes empty (`d`) – **destroys existing matter, and an `O` stays `O`**;
   * `hit == 1` (or 0) → unchanged (matter persists forever until destroyed/accepted).
5. Pattern detection (once per tick, after step 4):
   * every pattern `M` cell must hold matter (`M` or `m`);
   * every pattern `O` cell that is **hex-adjacent to a pattern `M` cell** ("skirt") must **not** hold matter;
   * other cells (`O` not adjacent to pattern matter, `-`, `*`) are don't-care.
   If it matches: pattern cells (only the pattern `M` cells) are removed and `accepted++`. At most **one acceptance per tick**.
* Hex adjacency in the text grid: `(dr,dc) ∈ {(0,±2),(±1,±1)}` (columns advance by 2 inside a row).
* Matter that is *not* in pattern ∪ skirt is irrelevant forever (never affects hits or detection). ⇒ **the relevant
  board state is only `M ∩ (pattern ∪ skirt)`** – use this for hashing/dedup.

### 2.2 Bitset form of a tick (for the fast simulator)
Per tick compute the 6 beam bitsets `b_e` (a line bitset, or 0 if emitter inactive/out of its window). Bit-sliced
counter (saturating):
```
c0=c1=c2=0
for b in beams: carry=c0&b; c0^=b; c2|= (c1&carry); c1^=carry
exact2 = c1 & ~c0 & ~c2
ge3    = (c1 & c0) | c2
M = (M | exact2) & ~ge3
if (M & P)==P and (M & Skirt)==0: M &= ~P; accepted += 1
```
`M` need not be masked to ∩(P∪Skirt) for correctness (junk outside is harmless) but mask it for hashing.
Interior cells ≤ 397 (R=12) ⇒ ≤ 7 `uint64` words per bitset.

### 2.3 What the example (reference) solution does **[VERIFIED]**
It only uses emitters 1 and 3 (one horizontal beam + one SW beam ⇒ **one crossing cell = a "pointer"**) and walks the
pointer over the pattern cell by cell; then repeats the same ~40–70-line loop K times. Per-copy cost ≈ 40 lines for a
5-cell pattern, 70 for an 8-cell pattern. It leaves 1–9 wasted trailing lines. Treat 446 (the example total) as the
"naive" level; a decent solution should be several times smaller.

---

## 3. Where the score is decided

Let `T1` = ticks to the first acceptance, `c` = steady-state ticks per additional copy. Total ≈ `T1 + (K-1)·c`
(never below `K`, since at most one acceptance per tick).

| file | dominant term | consequence |
|---|---|---|
| small (ΣK=58, tiny patterns) | both | exhaustive/near-exact search is affordable – do it, it also tests the engine |
| medium (ΣK=2045, ≤59 cells, R≤5) | **(K-1)·c** | every line shaved off `c` is worth up to ~2000 lines → **cycle finder is the highest-leverage module** |
| large (ΣK=689, R 6–8, ≤131 cells) | `T1` and `c` both | beam search + cycle reuse |
| enormous (K≤2, R 9–12, ≤297 cells) | **T1** | best raw build strategy; beam width/time is the lever |

Lower bounds to sanity-check results: `≥ K`; `T1 ≥ ⌈|P| / 12⌉`-ish (≤12 crossing cells per tick, a bit more with
coincident pairs, see §5.3); and `T1 ≥ min over axis pairs of (max line-offset distance) + 1`.

---

## 4. Geometry model **[VERIFIED]**

### 4.1 Border cycle and the "window"
* Border positions form a cycle of **6R** cells. Position 0 = emitter 1's corner (left-middle cell `(R,0)`), clockwise
  order = up the upper-left side → top edge → down the right side → bottom edge (right→left) → up the lower-left side.
  Emitter `e` (1..6) starts at position `(e-1)·R`. (Construction used in my verified simulators: `p=[(R,0)]`, then
  `(r,R-r)` for r=R-1..0; `(0,R+2j)` j=1..R; `(r,4R-|r-R|)` r=1..2R; `(2R,3R-2j)` j=1..R; `(r,|r-R|)` r=2R-1..R+1.)
* Row `r` (0..2R) has cells at columns `k+2j`, `k=|r-R|`, `j=0..2R-k`; the first and last cell of each row are border
  `*`, and rows 0 and 2R are entirely border. Interior cells = the rest (rows 1..2R-1; 397 cells for R=12).
* **Each emitter has exactly 2R-1 positions at which its beam crosses the interior: offsets `o ∈ [-(R-1), +(R-1)]`
  from its start position.** At offsets `±R` and beyond the beam runs along/outside the border (hits nothing =
  *dead*). Verified for R=3 and R=12 (e.g. R=12: 23 effective positions per emitter, chord length 23 at offset 0).
* Hence **every beam is a full chord of the hexagon along one of 3 axes** – never a partial ray.

### 4.2 Axial line indices (cell `(r,c)` lies on three lines)
```
H-line  = r            (axis of emitters 1 and 4, horizontal)
D-line  = c - r        (axis of emitters 2 and 5, "\" direction; indices have fixed parity, step 2)
E-line  = c + r        (axis of emitters 3 and 6, "/" direction; step 2)
```
Line index as a function of offset `o` (R=4 table printed by script; general form):

| emitter | axis | line index at offset o | F (+1) effect |
|---|---|---|---|
| 1 | H | `R - o` | row decreases (up) |
| 4 | H | `R + o` | row increases (down) |
| 2 | D | `R + 2o` | D-index increases |
| 5 | D | `R - 2o` | D-index decreases |
| 3 | E | `3R + 2o` | E-index increases |
| 6 | E | `3R - 2o` | E-index decreases |

(*Re-derive programmatically from the border-cycle + ray code instead of trusting the formulas – the table is only
for orientation.*) Key facts:
* **Both emitters of an axis can reach every line of that axis** and start on the central chord. So the two sliders of
  an axis are interchangeable *as line choosers*; they differ only in the sign of `F`. Canonicalise states by sorting
  the two line indices of an axis (≈ up to 8× fewer states).
* Offset `±R` is a free "silent parking" slot: an emitter that is **active but stepped out of its window emits
  nothing**, and stepping back in re-enables it. So an on/off switch can be realised by a *move* instead of a `T`
  (both cost one instruction slot, but the move also repositions). Keep positions mod 6R in the state so these
  cases are covered; positions beyond ±R are never useful (don't expand them).

### 4.3 Stamp view
For a configuration `s = (6 lines/offsets, 6 active flags)` define
`E2(s)` = cells hit exactly twice, `G3(s)` = cells hit ≥3 times. Then one tick is
`M ← (M ∪ E2(s)) \ G3(s)`. With 2 sliders per axis ≤ 12 crossing cells per tick (4 per axis pair) – fewer if some
coincide. **Coincident pair** (both sliders of an axis on the same line, both active) hits *every* cell of that
chord twice ⇒ **paints a whole chord (≤ 2R-1 cells) in one tick**. Any additional beam crossing that chord makes
that single cell a 3-hit ⇒ it is *not* created (and existing matter there is destroyed). That is the key tool for
"paint a line but leave holes/ends untouched".

---

## 5. Structural insights that drive the algorithm design

1. **Persistence.** Matter stays until a ≥3-hit or acceptance ⇒ the pattern can be assembled from stamps made on
   different ticks. Only the *union* of stamps matters (modulo destructions).
2. **Contamination rule.** The only forbidden M cells are the *skirt* (O cells adjacent to pattern matter; this
   includes holes inside the blob). Matter anywhere else is free. So a config is "safe" iff `E2(s) ∩ Skirt = ∅`
   (or the offending cells are 3-hit). Hard-pruning unsafe configs shrinks branching drastically.
3. **Pointer walk (baseline).** Two active sliders of different axes ⇒ exactly one crossing cell per tick (the
   other 4 inactive). Changing the two line indices by Δ∈{-1,0,1} each moves that cell; a walk staying inside the
   (connected) pattern never touches the skirt. ⇒ always-valid construction, cost ≈ `2|P|` + set-up.
4. **Parallelism is the whole game.** 6 sliders ⇒ up to 12 crossings/tick and chord-painting; the pointer walk uses
   1 cell/tick. Expect the optimised build to be several times faster than the baseline on blobs.
5. **Blocking.** To paint a chord with a coincident pair but keep chosen cells empty (skirt cells at the run ends,
   holes), put other beams through those cells in the same tick (3 hits). Every other beam crosses a given chord in ≤1
   cell, so ≤4 blockers per painted chord. Blockers of the D/E axes can track a run boundary that moves ±1 column
   per row by a single move per tick (D: stay=+1 col, step=-1 col; E mirrored) – this is the structure the search
   should be able to discover and that structured seeds (§8, M5) can exploit. **[HYPOTHESIS: valuable for dense
   blobs with few holes, e.g. the near-full-board R=11 pattern in `large`.]**
6. **No pipelining across copies.** Acceptance deletes every pattern cell, so each copy needs all |P| cells
   re-created; you cannot pre-build the next copy. Copies *can* share slider positions/activity: no re-toggling, no
   re-positioning → `c` is much smaller than `T1`.
7. **Ping-pong / palindrome cycle (guaranteed fallback for K>1)** **[HYPOTHESIS, needs simulation check]**: if the
   first build goes through configs `s1…sT` (accept at `sT`), the next copy can be `(no-op at sT) , s_{T-1}, …, s1`
   (i.e. reversed moves F↔B, toggles unchanged), the next `(no-op at s1), s2,…,sT`, … Each copy costs `T`
   ticks and needs no re-set-up. (Pure reversal without the extra no-op loses the stamp of the final config.)
   Order of stamps changes only matters if ≥3-hit destruction interacts → re-verify by simulation.
8. **Static re-creation (c=1).** If the final config's `E2` already covers all of P and is safe, copies cost 1 line
   each (a no-op line re-creates and re-accepts). Possible for very small/regular patterns; found automatically by
   the cycle finder (cycle length 1).

---

## 6. Algorithm plan

### M0 – Core library (do this first, test heavily)
* Parse input; build per-radius tables once: border cycle, per `(emitter, position)` ray bitset over interior cells,
  `lineBits[axis][lineIndex]`, interior-cell indexing, adjacency lists.
* Fast simulator `simulate(program, pattern, K) → (accepted ticks, final state)` in the bitset form of §2.2.
* **Reference re-implementation of the sandbox JS semantics in plain code** (slow, string/grid based) and a
  differential test: random programs on random patterns, fast sim == slow sim tick-by-tick (accepted tick list +
  relevant board). Also replay `out_level-6_0-example.txt`: must give acceptance ticks `[29,69,109] / [55] /
  [55,125,195,265]`.
* Output writer replicating the exact example format. Final validator: re-parse the written file and simulate with
  the *slow* reference sim; assert accepted == K exactly for every program, and report total lines.

### M1 – Baseline (guaranteed valid, also the incumbent upper bound)
* Choose two emitters of different axes (e.g. 1 and 2, try all 3 axis pairs × both emitter choices, keep the
  shortest). Pointer cell = intersection of their lines. Express pattern cells in (line_a, line_b) coordinates.
* Route = shortest walk that visits all pattern cells using steps with Δ ∈ {-1,0,1}² in line-offset space
  (hex neighbours + 2-apart jumps along the 3rd axis), staying inside P (so no skirt contamination):
  DFS/Euler walk of a spanning tree, then improve with nearest-neighbour + 2-opt style shortcuts, or a small DP for ≤ 12 cells.
* Set-up: move both sliders (inactive) to the first cell, toggle both on the same tick as arrival (the toggle tick
  already creates the first cell). K copies: use ping-pong (§5.7) – walk forward, then backward through the same
  cells (with the extra no-op at the turn).
* Truncate the program at the K-th acceptance. Expected ≈ `2|P|·…` lines for the first copy, much better than the
  reference but far from optimal – **its job is correctness and an upper bound for pruning.**

### M2 – Main engine: beam search over 6-slider configurations **[HYPOTHESIS on parameters, design recommended]**
* **State**: `(offsets/positions of 6 emitters, 6 active flags, relevant-board bits M∩(P∪Skirt), accepted count)` +
  backpointer (parent index, action) for path reconstruction. Canonicalise per-axis pairs for the hash key
  (keep real emitter identities in the stored action; canonicalisation is only for dedup – when expanding, expand
  the canonical representative with both emitter assignments handled correctly, or simply do not canonicalise at
  first and measure).
* **Actions**: per emitter `{-,F,B,T}` → ≤ 4096 joint actions. Prune:
  * `F/B` that would leave `[-R, R]` offset range;
  * toggling/moving an inactive, parked, irrelevant emitter combinations that are provably equivalent (e.g. two
    consecutive moves of a dead slider);
  * **unsafe successors** (`E2(s') ∩ Skirt ≠ ∅` not cancelled by `G3`) – hard prune (optionally allow in a relaxed
    second pass with a penalty, because erasing contamination costs ticks);
  * successors that destroy already-built pattern cells (`G3 ∩ M ∩ P ≠ ∅`) – prune or penalise.
  Cost per successor is a few hundred bit-ops; with numba/native code ~0.1–0.3 µs, i.e. ≈ 1 ms to expand one state
  fully.
* **Layered beam**: all states in layer t have exactly t ticks. Keep the best `W` by score, dedup by hash of
  (canonical config, masked board, accepted count). Stop when a state has accepted K copies; prune any state whose
  `t + h ≥ incumbent` (incumbent = baseline or best beam result so far ⇒ branch-and-bound flavour).
* **Score / heuristic `h`** (lower better; same `t` inside a layer):
  * `missing` = |P \ M|, `contam` = |Skirt ∩ M|.
  * `reach(p)` for each missing cell p with line ids `(aH,aD,aE)`: for every axis pair `(X,Y)` the number of moves
    the nearest slider of X needs to reach `aX` (+1 if it is inactive and must be toggled), same for Y,
    `d = min over pairs of max(costX,costY)`; `reach = max_p d(p)` (precompute per-axis-line distance tables).
  * `h = max(reach, ⌈missing / κ⌉)` with κ≈6–8 (tune) `+ λ·contam` (λ≈0.5–1, tune).
  * Tie-break: more pattern cells already present, then more active sliders near targets, then random jitter.
  * Add **diversity control**: cap states per canonical config (e.g. ≤ 4–8) so the beam doesn't collapse to one
    trajectory family.
* **Multi-copy mode**: same search, objective = ticks until accepted==K. Rank by `(accepted desc, then h)`.
  For K>2 run it only up to `K' = min(K, 3 or 4)` copies and extract the periodic tail (§M3) – do not run 19 copies.
* **Width schedule**: start `W=200` (fast, finds a valid solution), then rerun with `W = 1k, 5k, 20k…` with the
  incumbent bound, as time allows; keep the best. Use all cores: one pattern per process (`multiprocessing`),
  order jobs largest-first.

### M3 – Cycle finder for K>1 (highest leverage on `medium`)
Goal: the program = `prefix` (reach the steady state and make copy 1) + `(K-1)` repetitions of a loop segment of
length `c` (each producing one acceptance), possibly truncated at the final acceptance.
1. From multi-copy beam results (K'≥3), detect periodicity: states at consecutive acceptance ticks (slider config +
   masked board) repeating ⇒ loop found → extrapolate to K.
2. Dedicated **closed-walk search**: take candidate start configs `s0` (final configs of the best first-copy
   solutions, plus top-N configs ranked by `|E2(s) ∩ P|`), run a beam/IDDFS of depth `c = 1,2,3,…` that must
   (a) rebuild P from an empty pattern region, (b) accept, (c) return to `s0`. Smallest `c` wins. For depth ≤ ~6 an
   exhaustive IDDFS over *safe* configs with the bound "uncovered cells ≤ remaining ticks × max stamp size" is
   feasible for R ≤ 5.
3. Fallback: **ping-pong** from §5.7 (cost `T_core` per copy, no set-up) and the **naive repeat** of the whole
   first-copy program only if it returns to its own start state (it usually doesn't).
4. Choose the cheapest among {direct K-copy beam result, prefix + loops, ping-pong, baseline}; always re-verify the
   final program by full simulation.

### M4 – Post-optimisation (cheap, always run)
* Cut every line after the K-th acceptance.
* **Line deletion**: try removing each line (and merging adjacent lines by combining their non-conflicting
  instructions: two lines whose same-emitter slots are not both used can sometimes merge), re-simulate, keep if still
  K acceptances. Also try cancelling `F` followed by `B` for the same emitter when no hit-relevant tick lies between.
* **Window shifting / local search**: perturb a segment (random ± moves/toggles), re-simulate, accept if not longer
  and still valid; use as a time-boxed LNS (large neighbourhood: re-run the beam between two states of an existing
  solution with a tighter bound).

### M5 – Optional stronger ideas (only after M0–M4 work and are measured)
* **Structured sweep seeds**: painter pair (coincident sliders) sweeping the chords of the best axis one per tick,
  plus ≤4 tracker sliders (the other axes' sliders) acting as blockers on run ends / holes (§5.5). Generate the
  seed program by DP over (chord order, tracker assignment), then hand it to M4/LNS or inject it into the beam as
  a start state. Most useful for `large`/`enormous`.
* **Exact solver for tiny patterns** (≤ ~8 cells): A*/IDA* over the stamp-config graph with an admissible reach
  heuristic to certify optimal lengths and measure the beam's quality gap.
* **Erase planning**: if contamination is unavoidable, add an explicit "erase" move (3-hit on a skirt cell using a
  coincident pair + a crossing beam, or three concurrent distinct-axis beams).
* Symmetry: the hexagon has D6 symmetry; a pattern may map onto a rotated/reflected pattern with a better-conditioned
  sliders assignment (rotate emitter roles) – only useful if searches turn out asymmetric in quality.

---

## 7. Data structures / performance notes

* Language: **no C/C++ compiler is installed on this machine** (g++/gcc/clang/rustc/go absent). Available: Python
  3.13 (numpy 2.4 present; `pip` works – `numba` was downloadable, install it), Node.js, `dotnet` runtime only (no SDK
  detected), 12 logical cores. Recommended: **Python + numba (`@njit`) kernels** for simulation/beam inner loops +
  `multiprocessing` over patterns; or, if easy, install a native toolchain (e.g. WinLibs/LLVM via `winget`) and write
  the engine in C++/Rust. Pure-Python beam search would be ~100× too slow.
* Bitsets: ≤7 `uint64` words (R=12). Precompute `lineBits[axis][idx]` and the ray bitsets per (emitter, offset).
  Per successor: 6 lookups + counter logic ≈ 100–200 ops.
* Zobrist/xor-shift hash of (positions, flags, masked board words) for dedup; beam arrays as struct-of-arrays;
  store parent pointers per layer (reconstruct instruction lines at the end).
* Rough compute budget **[ESTIMATE]**: expansions ≈ states × ~1–4k actions × ~0.2 µs ≈ 0.2–1 ms per state ⇒ `W=1000`,
  `T=150` ⇒ ~1–2 min/pattern single core for the enormous set; 140 patterns / 12 cores ⇒ ~15–30 min per width step.
  Small/medium patterns are orders of magnitude cheaper (R≤5, T≤30 per copy).
* Keep every result on disk per pattern (best program + score) so runs are resumable and improvements are monotone
  ("never replace a program by a longer valid one").

---

## 8. Per-file strategy and time allocation

| file | plan |
|---|---|
| example/small | run everything incl. exact search for tiny ones; use them to tune heuristic weights and verify cycle logic. Compare to reference (446 lines for example). |
| medium | cycle finder is the priority. Run multi-copy beam (K'≤4) + closed-walk search for every pattern; budget most CPU here relative to its size; the payoff scales with K. |
| large | beam W≥1k for first build + cycle reuse (K≤6); apply M4/M5 seeds. |
| enormous | pure first-build optimisation (K≤2): biggest beam widths, longest time slice, M5 sweep seeds; second copy via ping-pong/cycle. |

Order of work: M0 → M1 (all five files valid, baseline score table) → M2 on small/medium → M3 → M2 on large/enormous
→ M4 everywhere → M5 as time permits. Always keep the best-known valid solution file per input and re-export after
each improvement round (a valid file at every stage).

---

## 9. Validation & pitfalls checklist

* Differential test fast sim vs. slow reference sim; replay example output; also test against the real sandbox by
  pasting a few generated programs (it shows `Accepted x / K` and "✓ Valid").
* **Exactly K acceptances; last line = the K-th acceptance tick** (reference solution wastes lines after it; also
  check the program never accepts *earlier than expected in a way that overshoots K*).
* Instruction semantics: a single emitter cannot `F` and `T` in the same line. `T` on an emitter currently parked at
  `±R` or beyond still flips the flag.
* Cleanup timing: matter accepted on tick t can be re-made on tick t+1 (cleanup precedes beams) – relevant for
  `c=1` and ping-pong cycles; the destroyed (`d`) cell and accepted (`x`) cell are `O` for the next tick's beams.
* `hit>=3` destroys *existing* matter including pattern cells already built – the safe-config filter must also
  consider `G3 ∩ (M ∩ P)`.
* Skirt includes **holes inside the blob** and cells that are adjacent to the pattern *only diagonally in the hex
  sense* (all 6 hex neighbours). Compute it exactly as in `testPattern` (neighbours via the 6 `DIR_TO` offsets on the
  pattern text grid, restricted to cells that are `O`).
* Output format: tokens single-space separated, one blank line between programs, same trailing newline convention as
  `out_level-6_0-example.txt`. Name outputs `out_level-6_1-small.txt`, `…_2-medium.txt`, `…_3-large.txt`,
  `…_4-enormous.txt` (and example) in the `Task 6` folder.
* Determinism: seed all randomness; log per-pattern score so regressions are visible.

---

## 10. Expected outcome ranges **[ESTIMATES – to be confirmed by measurement]**

* Baseline pointer walk: roughly `2|P| + R` lines for the first copy (enormous ≈ 150–600 lines per pattern).
* Beam search: should reduce the first build several-fold on dense blobs because chord-painting/parallel crossings
  create many cells per tick; the example 5-cell pattern should need ~10–15 lines for the first copy and a
  small `c` (2–6) afterwards, versus the reference's 29 + 2×40.
* The final score is dominated by `medium`'s Σ(K-1)·c and the `large`/`enormous` build times; track these as the KPIs.

---

## 11. Experiments / decision checkpoints for the implementer

1. **Sim equivalence** (M0): must pass before anything else. If the replay of the reference output doesn't give
   `[29,69,109] / [55] / [55,125,195,265]`, stop and fix semantics.
2. **Baseline score table** per file (lines per pattern vs |P|, R, K).
3. **Beam heuristic tuning** on `small` + first 30 `medium` patterns: sweep `κ, λ, W, diversity cap`; keep a table of
   total lines vs. runtime. If hard-pruning unsafe configs yields no solution for some pattern → enable the relaxed
   mode with erase moves.
4. **Cycle value**: for 20 medium patterns compare (a) K×naive, (b) ping-pong, (c) closed-walk `c`. Decide how much CPU
   the closed-walk search deserves.
5. **Structured sweep check (M5)**: on the 3–5 densest `large`/`enormous` patterns, hand-count how many skirt cells
   each chord needs blocked (≤4 is feasible); decide if seeds are worth coding.
6. **Time boxing**: after every full pass, print current total per file; stop when improvement per CPU-hour flattens.

### Open questions (cannot be settled from the files)
* Whether the judge accepts *more* than K acceptances (we avoid the question by stopping at K).
* Whether score is the raw line total (as stated in the PDF) or relative to thresholds – irrelevant to the method
  but affects how long to keep optimising.
