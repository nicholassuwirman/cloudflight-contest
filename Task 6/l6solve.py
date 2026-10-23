"""Level 6 solver driver."""
import numpy as np
from l6core import simulate_ref
from l6model import PatternModel
from l6beam import beam_search


def actions_to_program(path, sign):
    prog = []
    for act in path:
        line = []
        for e in range(6):
            u = (int(act) >> (2 * e)) & 3
            if u == 0:
                line.append('-')
            elif u == 3:
                line.append('T')
            elif u == 1:
                line.append('F' if sign[e] == 1 else 'B')
            else:
                line.append('B' if sign[e] == 1 else 'F')
        prog.append(line)
    return prog


DEFAULT = dict(W=300, CAP=3000, iB=4, iP=1, wS=0.05, wM=0.5, seed=1, jit=0)


def solve_beam(case, pm=None, maxT=None, **kw):
    p = dict(DEFAULT)
    p.update(kw)
    if pm is None:
        pm = PatternModel(case)
    R = case['R']
    if maxT is None:
        maxT = 60 + 8 * pm.nP * case['K'] // 4 + 4 * R
    L0 = np.full(6, R, dtype=np.int64)
    A0 = np.zeros(6, dtype=np.int64)
    b0 = np.zeros(pm.NW, dtype=np.uint64)
    depth, path = beam_search(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, case['K'],
                              L0, A0, b0, 0, p['W'], p['CAP'], maxT,
                              p['iB'], p['iP'], p['wS'], p['wM'], p['seed'], p['jit'])
    if depth < 0:
        return None
    prog = actions_to_program(path, pm.sign)
    acc = simulate_ref(case, prog)
    assert len(acc) == case['K'] and acc[-1] == len(prog), (acc, len(prog))
    return prog


def solve(case, **kw):
    return solve_beam(case, **kw)
