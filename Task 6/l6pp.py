"""Ping-pong / segment-reuse edges for the copy graph (plan.md section 5.7, improved-plan-1 section 3).

For every known copy segment X -> Y (configs s_0 = X, ..., s_c = Y) candidate segments from Y are
  reverse:        inv(a_c), ..., inv(a_1)            (visits s_{c-1}, ..., s_0)
  noop + reverse: -, inv(a_c), ..., inv(a_1)         (also re-stamps s_c first)
  replay:         a_1, ..., a_c  and  -, a_1, ..., a_c   (same moves from Y)
each truncated at its first acceptance (exact simulation from Y with an empty board, skirt matter tracked).
They are added to a CopyGraph seeded with the incumbent; the exact K-edge DP picks the best combination, the
segments of the new best path generate more candidates, repeat. The result is verified with simulate_ref.
"""
from l6core import simulate_ref
from l6graph import CopyGraph, program_to_actions, apply_cfg, swap_mask, swap_act, INF

INV = (0, 2, 1, 3)


def inv_act(a):
    r = 0
    for e in range(6):
        r |= INV[(a >> (2 * e)) & 3] << (2 * e)
    return r


class SegSim:
    def __init__(self, pm):
        self.nl = pm.lm.nl
        NW = pm.NW
        self.lb = [[sum(int(pm.lineBits[a, l, w]) << (64 * w) for w in range(NW)) for l in range(self.nl)]
                   for a in range(3)]
        self.Pm = sum(int(pm.Pm[w]) << (64 * w) for w in range(NW))
        self.Sm = sum(int(pm.Sm[w]) << (64 * w) for w in range(NW))

    def first_accept(self, cfg, acts):
        """1-based tick of the first acceptance when acts run from cfg on an empty board, else None."""
        L, A = list(cfg[0]), list(cfg[1])
        lb, Pm, Sm, nl = self.lb, self.Pm, self.Sm, self.nl
        M = 0
        for t, act in enumerate(acts):
            for e in range(6):
                u = (act >> (2 * e)) & 3
                if u == 1:
                    L[e] += 1
                    if L[e] >= nl:
                        return None
                elif u == 2:
                    L[e] -= 1
                    if L[e] < 0:
                        return None
                elif u == 3:
                    A[e] ^= 1
            c0 = c1 = c2 = 0
            for e in range(6):
                if A[e]:
                    b = lb[e % 3][L[e]]
                    carry = c0 & b
                    c0 ^= b
                    c2 |= c1 & carry
                    c1 ^= carry
            M = (M | (c1 & ~c0 & ~c2)) & ~((c1 & c0) | c2)
            if (M & Pm) == Pm and not (M & Sm):
                return t + 1
        return None


def segment_candidates(cfg, acts):
    inv = [inv_act(a) for a in reversed(acts)]
    return [inv, [0] + inv, list(acts), [0] + list(acts)]


def add_pp_edges(g, sim, segs, seen):
    new = []
    for cfg, acts in segs:
        end = cfg
        for a in acts:
            end = apply_cfg(end, a)
        for seq in segment_candidates(cfg, acts):
            key = (end, tuple(seq))
            if key in seen:
                continue
            seen.add(key)
            r = sim.first_accept(end, seq)
            if r is None:
                continue
            s = seq[:r]
            to = end
            for a in s:
                to = apply_cfg(to, a)
            g.add_edge_exact(end, s, to)
            new.append((end, s))
    return new


def path_segments(g, edges):
    """Exact-frame (cfg_from, acts) of each edge on a DP path."""
    cfg = g.start
    out = []
    for e in edges:
        s = g.esrc[e]
        m = swap_mask(g.nodes[s], cfg)
        acts = [swap_act(a, m) for a in g.eacts[e]]
        out.append((cfg, acts))
        for a in acts:
            cfg = apply_cfg(cfg, a)
    return out


def solve_pingpong(case, incumbent, rounds=8, graph=None):
    if case['K'] < 2:
        return None
    g = graph or CopyGraph(case)
    g.seed_program(incumbent)
    sim = SegSim(g.pm)
    seen = set()
    acts = program_to_actions(incumbent, g.pm.sign)
    acc = simulate_ref(case, incumbent)
    segs = []
    cfg, prev = g.start, 0
    for t in acc:
        seg = acts[prev:t]
        segs.append((cfg, seg))
        for a in seg:
            cfg = apply_cfg(cfg, a)
        prev = t
    best_total, best_prog = len(incumbent), None
    for _ in range(rounds):
        new = add_pp_edges(g, sim, segs, seen)
        if not new:
            break
        f, src, dst, w = g.dp()
        edges, total = g.best_path(f, src, dst, w)
        segs = new + (path_segments(g, edges) if edges else [])
        if total >= best_total:
            continue
        prog = g.build_program(edges)
        a2 = simulate_ref(case, prog)
        assert len(a2) == case['K'] and a2[-1] == len(prog), (a2, len(prog), total)
        best_total, best_prog = total, prog
    return best_prog
