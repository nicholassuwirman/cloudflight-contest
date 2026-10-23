"""Paired T1 bench (first build only, K forced to 1) for beam variants.

usage: python bench_t1.py <tag> <n> <procs> '<json list of variant dicts>'
  variant keys: fn (beam|beam2), W, CAP, iB, iP, wS, wM, wL, LX, seed, sym ([k,m])
Patterns: the n densest (nP / (nP + nS)) of the file.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
from l6core import parse_input
from l6model import PatternModel

BASE = dict(fn='beam', W=200, iB=4, iP=1, wS=0.05, wM=0.5, wL=0.0, LX=0, seed=1, sym=[0, 0], lockAx=-1, lockT=0, wC=0.0)


def t1(args):
    case, v = args
    from l6sym import transform_case
    p = dict(BASE)
    p.update(v)
    k, m = p['sym']
    q = transform_case(case, k, m) if (k, m) != (0, 0) else case
    pm = PatternModel(q)
    R = q['R']
    L0 = np.full(6, R, dtype=np.int64)
    A0 = np.zeros(6, dtype=np.int64)
    b0 = np.zeros(pm.NW, dtype=np.uint64)
    W = p['W']
    CAP = p.get('CAP', 10 * W)
    maxT = 200
    t0 = time.time()
    if p['fn'] == 'beam':
        from l6beam import beam_search
        d, _ = beam_search(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, 1, L0, A0, b0, 0,
                           W, CAP, maxT, p['iB'], p['iP'], p['wS'], p['wM'], p['seed'], 0)
    else:
        from l6beam2 import beam_search2
        d, _ = beam_search2(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, 1, L0, A0, b0, 0,
                            W, CAP, maxT, p["iB"], p["iP"], p["wS"], p["wM"], p["seed"], 0, p["wL"], p["LX"], p["lockAx"], p["lockT"], p["wC"])
    return d, time.time() - t0


def main():
    tag, n, procs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variants = json.loads(sys.argv[4])
    cs = parse_input(f'in_level-6_{tag}.txt')
    dens = []
    for i, c in enumerate(cs):
        pm = PatternModel(c)
        dens.append((pm.nP / (pm.nP + pm.nS), i))
    idx = [i for _, i in sorted(dens, reverse=True)[:n]]
    print('patterns', idx, flush=True)
    with Pool(procs) as pool:
        for v in variants:
            t0 = time.time()
            res = pool.map(t1, [(cs[i], v) for i in idx])
            ds = [r[0] for r in res]
            print(f'{json.dumps(v)}: sumT1={sum(ds)} cpu={sum(r[1] for r in res):.0f}s wall={time.time() - t0:.0f}s',
                  ' '.join(map(str, ds)), flush=True)


if __name__ == '__main__':
    main()
