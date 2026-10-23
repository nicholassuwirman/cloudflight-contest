"""D6 symmetric restarts (improved-plan-1 §5) and the portfolio solver.

solve(g(pattern)) -> map the program back with the emitter permutation of g (F/B swapped for reflections)
-> verified with simulate_ref on the original pattern.
"""
import time
from l6core import geom, FIRE, simulate_ref
from l6solve import solve_beam, DEFAULT


def _rot(R, r, c):
    z = r - R
    x = (c - 2 * R - z) // 2
    x, z = -z, -(-x - z)          # (x,y,z) -> (-z,-x,-y)
    return z + R, 2 * x + z + 2 * R


def sym_map(R, k, m):
    """Cell map for symmetry (k rotations after optional vertical reflection m)."""
    def f(r, c):
        if m:
            r = 2 * R - r
        for _ in range(k):
            r, c = _rot(R, r, c)
        return r, c
    return f


_CACHE = {}


def sym_info(R, k, m):
    """Returns (cell map, perm) where original emitter perm[j] plays transformed emitter j."""
    key = (R, k, m)
    if key in _CACHE:
        return _CACHE[key]
    g = geom(R)
    f = sym_map(R, k, m)
    perm = [None] * 6
    for i in range(6):
        sr, sc = g.perimeter[i * R]
        dr, dc = FIRE[i]
        tr, tc = f(sr, sc)
        er, ec = f(sr + dr, sc + dc)
        for j in range(6):
            if g.perimeter[j * R] == (tr, tc) and FIRE[j] == (er - tr, ec - tc):
                perm[j] = i
    assert None not in perm, (R, k, m, perm)
    _CACHE[key] = (f, perm)
    return f, perm


def transform_case(case, k, m):
    R = case['R']
    f, _ = sym_info(R, k, m)
    rows, cols = case['rows'], case['cols']
    out = [['-'] * cols for _ in range(rows)]
    for r in range(rows):
        off = abs(r - R)
        for c in range(off, cols - off, 2):
            r2, c2 = f(r, c)
            out[r2][c2] = case['grid'][r][c]
    q = dict(case)
    q['grid'] = [''.join(row) for row in out]
    return q


FLIP = {'F': 'B', 'B': 'F', 'T': 'T', '-': '-'}


def map_back(prog, perm, m):
    out = []
    for line in prog:
        nl = ['-'] * 6
        for j in range(6):
            tok = line[j]
            nl[perm[j]] = FLIP[tok] if m else tok
        out.append(nl)
    return out


SYMS = [(k, m) for m in (0, 1) for k in range(6)]

# parameter portfolio (from the tune sweep: no single set dominates)
PORTFOLIO = [
    dict(iB=4, iP=1, wS=0.05, wM=0.5),
    dict(iB=2, iP=1, wS=0.05, wM=2.0),
    dict(iB=4, iP=0, wS=0.2, wM=2.0),
    dict(iB=4, iP=1, wS=0.0, wM=0.5),
]


def solve_sym(case, incumbent=None, budget=60.0, W=800, CAP=None, nsym=12, nport=1, seed=1, verbose=False, **kw):
    t0 = time.time()
    best = incumbent
    K = case['K']
    runs = []
    for pi in range(nport):
        for si in range(nsym):
            runs.append((SYMS[si], PORTFOLIO[pi % len(PORTFOLIO)], seed + pi))
    for (k, m), par, sd in runs:
        if time.time() - t0 > budget:
            break
        q = transform_case(case, k, m)
        p = dict(par)
        p.update(kw)
        p['W'] = W
        p['CAP'] = CAP or 10 * W
        p['seed'] = sd
        maxT = len(best) if best is not None else None
        try:
            prog = solve_beam(q, maxT=maxT, **p)
        except AssertionError:
            prog = None
        if prog is None:
            continue
        _, perm = sym_info(case['R'], k, m)
        back = map_back(prog, perm, m)
        acc = simulate_ref(case, back)
        assert len(acc) == K and acc[-1] == len(back), ('symmetry map failed', k, m, acc, len(back))
        if verbose:
            print(f'  sym k={k} m={m} {par} -> {len(back)}')
        if best is None or len(back) < len(best):
            best = back
    return best
