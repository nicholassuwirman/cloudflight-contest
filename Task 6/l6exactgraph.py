"""Copy graph with CP-SAT edges (improved-plan-2 sections 6.3 and 8).

Node expansion = exact shortest one-copy segments from the node's config (empty board): the smallest c for
which the CP-SAT segment model is feasible, plus up to `nsol` alternative end configs at that length (and at
c+1 when `extra`), found by excluding each found end config with a no-good. Edges are merged into a CopyGraph
that already holds the incumbent's segments (and their ping-pong variants); the exact K-edge DP picks the path.
"""
import time
import numpy as np
from l6core import simulate_ref
from l6exact import Segment
from l6graph import CopyGraph, canon_key, INF
from l6pp import SegSim, add_pp_edges


def exact_edges(case, pm, cfg, cmin, cmax, nsol, tl, workers, deadline, extra=False):
    """[(acts, end_cfg)] of the shortest feasible c in [cmin, cmax] (+ c+1 if extra); [] if none / no time."""
    out = []
    found_c = None
    for c in range(cmin, cmax + 1):
        if time.time() > deadline:
            break
        seg = Segment(case, c, start=cfg, pm=pm)
        n_here = 0
        for _ in range(nsol):
            left = deadline - time.time()
            if left <= 0:
                break
            st = seg.solve(tl=min(tl, left), workers=workers)
            if st != 'FEASIBLE':
                break
            acts = seg.actions()
            end = seg.config(c)
            out.append((acts, end))
            n_here += 1
            # exclude this end config (exact; the canonical swap is handled by the graph)
            lits = [seg.pos[e][c][end[0][e]].Not() for e in range(6)]
            lits += [seg.A[e][c].Not() if end[1][e] else seg.A[e][c] for e in range(6)]
            seg.m.AddBoolOr(lits)
        if n_here:
            found_c = c
            break
    if extra and found_c is not None and found_c + 1 <= cmax and time.time() < deadline:
        seg = Segment(case, found_c + 1, start=cfg, pm=pm)
        for _ in range(max(1, nsol // 2)):
            left = deadline - time.time()
            if left <= 0:
                break
            if seg.solve(tl=min(tl, left), workers=workers) != 'FEASIBLE':
                break
            acts, end = seg.actions(), seg.config(found_c + 1)
            out.append((acts, end))
            lits = [seg.pos[e][found_c + 1][end[0][e]].Not() for e in range(6)]
            lits += [seg.A[e][found_c + 1].Not() if end[1][e] else seg.A[e][found_c + 1] for e in range(6)]
            seg.m.AddBoolOr(lits)
    return out


def solve_exact_graph(case, incumbent, budget=120.0, nsol=4, tl=20.0, workers=4, cmin=1, t1min=1,
                      extra=True, verbose=False):
    t0 = time.time()
    deadline = t0 + budget
    K = case['K']
    g = CopyGraph(case)
    pm = g.pm
    g.log = []
    sim = SegSim(pm)
    seen = set()

    def pp_round():
        for _ in range(3):
            segs, g.log = g.log, []
            if not segs:
                break
            add_pp_edges(g, sim, segs, seen)
        g.log = []
    g.seed_program(incumbent)
    pp_round()
    acc = simulate_ref(case, incumbent)
    T1 = acc[0]
    gaps = [acc[j] - acc[j - 1] for j in range(1, len(acc))]
    gmax = max(gaps) if gaps else T1
    best_total, best_prog = len(incumbent), None
    expanded = set()

    def expand(nid, lo, hi):
        expanded.add(nid)
        cfg = g.nodes[nid]
        for acts, end in exact_edges(case, pm, cfg, lo, hi, nsol, tl, workers, deadline, extra):
            g.add_edge_exact(cfg, acts, end)
        pp_round()

    expand(0, t1min, T1 - 1 if K == 1 else T1)
    while True:
        f, src, dst, w = g.dp()
        edges, total = g.best_path(f, src, dst, w)
        if edges is not None and total < best_total:
            prog = g.build_program(edges)
            a2 = simulate_ref(case, prog)
            assert len(a2) == K and a2[-1] == len(prog), (a2, len(prog), total)
            best_total, best_prog = total, prog
            if verbose:
                print(f'  exact graph: {total} ({time.time() - t0:.0f}s)', flush=True)
        if K == 1 or time.time() > deadline:
            break
        # next node: reachable, unexpanded, smallest optimistic completion f[k][X] + (K-k) * cmin
        ks = np.arange(K)[:, None]
        pr = (f[:K] + (K - ks) * cmin).min(axis=0).astype(np.float64)
        pr[f[:K].min(axis=0) >= INF] = np.inf
        for nid in expanded:
            pr[nid] = np.inf
        nid = int(np.argmin(pr))
        if not np.isfinite(pr[nid]) or pr[nid] >= best_total:
            break
        expand(nid, cmin, gmax)
    if verbose:
        print(f'exact graph: nodes={len(g.nodes)} edges={len(g.esrc)} exp={len(expanded)} total={best_total} '
              f'{time.time() - t0:.0f}s', flush=True)
    return best_prog
