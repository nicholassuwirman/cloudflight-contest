"""Experiment: suffix re-planning. Replay the best program to tick t, re-run the beam from that exact state
(config, built cells, acceptance count) with a wider beam, splice, verify.

usage: python exp_suffix.py <tag> <step> <W> [fractions]
"""
import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
from l6core import parse_input, simulate_ref
from l6model import PatternModel
from l6graph import program_to_actions
from l6solve import actions_to_program
from l6beam import beam_search


def state_at(pm, acts, t, R):
    L = [R] * 6
    A = [0] * 6
    NW = pm.NW
    lb = pm.lineBits
    M = np.zeros(NW, dtype=np.uint64)
    acc = 0
    for act in acts[:t]:
        for e in range(6):
            u = (act >> (2 * e)) & 3
            L[e] += 1 if u == 1 else (-1 if u == 2 else 0)
            A[e] ^= 1 if u == 3 else 0
        cnt = [np.zeros(NW, dtype=np.int64) for _ in range(0)]
        c0 = np.zeros(NW, dtype=np.uint64); c1 = c0.copy(); c2 = c0.copy()
        for e in range(6):
            if A[e]:
                b = lb[e % 3, L[e]]
                carry = c0 & b
                c0 = c0 ^ b
                c2 = c2 | (c1 & carry)
                c1 = c1 ^ carry
        e2 = c1 & ~c0 & ~c2
        g3 = (c1 & c0) | c2
        M = (M | e2) & ~g3
        assert not (M & pm.Sm).any()
        if ((M & pm.Pm) == pm.Pm).all():
            M = M & ~pm.Pm
            acc += 1
    return np.array(L, dtype=np.int64), np.array(A, dtype=np.int64), (M & pm.Pm), acc


def work(args):
    tag, i, case, prog, W, fracs = args
    pm = PatternModel(case)
    acts = program_to_actions(prog, pm.sign)
    best = list(acts)
    t0 = time.time()
    for f in fracs:
        t = int(len(acts) * f)
        L0, A0, b0, acc0 = state_at(pm, best, t, case['R'])
        for (iB, iP, wS, wM, sd) in [(4, 1, 0.05, 0.5, 1), (2, 1, 0.05, 2.0, 2)]:
            maxT = len(best) - t
            d, path = beam_search(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, case['K'],
                                  L0, A0, b0, acc0, W, 10 * W, maxT, iB, iP, wS, wM, sd, 0)
            if d >= 0 and t + d < len(best):
                cand = best[:t] + [int(x) for x in path]
                p = actions_to_program(cand, pm.sign)
                a = simulate_ref(case, p)
                if len(a) == case['K'] and a[-1] == len(p):
                    best = cand
    return tag, i, len(prog), len(best), time.time() - t0


def main():
    tag, step, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fracs = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [0.3, 0.5, 0.7]
    cs = parse_input(f'in_level-6_{tag}.txt')
    b = {int(k): v for k, v in json.load(open(f'best/{tag}.json')).items()}
    jobs = [(tag, i, cs[i], b[i], W, fracs) for i in range(0, len(cs), step)]
    o = n = 0
    with Pool(os.cpu_count()) as pool:
        for tg, i, old, new, dt in pool.imap_unordered(work, jobs):
            o += old; n += new
            print(i, old, new, f'{dt:.1f}s', flush=True)
    print('total', o, '->', n)


if __name__ == '__main__':
    main()
