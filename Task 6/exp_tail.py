"""Experiment: free-ended tail repair with CP-SAT (improved-plan-2 section 6.4).

For the last copy of the incumbent: state at t = len - r0 (config + built P cells, skirt clean), ask CP-SAT for an
acceptance within r < r0 ticks (r = r0-1, r0-2, ... while feasible). Reports the gain per pattern.

usage: python exp_tail.py <tag> <step> <r0> [timelimit] [workers]
"""
import sys
import time
import numpy as np
from l6core import parse_input, simulate_ref
from l6model import PatternModel
from l6graph import program_to_actions
from l6solve import actions_to_program
from l6exact import Segment
from improve import state_at
from run import load_best


def tail_repair(case, prog, r0, tl, workers, pm=None):
    pm = pm or PatternModel(case)
    acts = program_to_actions(prog, pm.sign)
    t = len(acts) - r0
    acc = simulate_ref(case, prog)
    if t < 0 or (len(acc) >= 2 and t < acc[-2]):
        return None, 'window crosses an acceptance'
    L, A, b, acc0 = state_at(pm, acts, t, case['R'])
    built0 = {q for q in range(pm.nP) if (int(b[q >> 6]) >> (q & 63)) & 1}
    best = None
    log = []
    for r in range(r0 - 1, 0, -1):
        seg = Segment(case, r, start=(L, A), built0=built0, pm=pm)
        seg.m.AddHint  # noqa
        st = seg.solve(tl=tl, workers=workers)
        log.append((r, st, round(seg.time, 1)))
        if st != 'FEASIBLE':
            break
        cand = acts[:t] + seg.actions()
        p = actions_to_program(cand, pm.sign)
        a = simulate_ref(case, p)
        assert len(a) == case['K'] and a[-1] == len(p), (a, len(p))
        best = p
    return best, log


def main():
    tag, step, r0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 30
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    cs = parse_input(f'in_level-6_{tag}.txt')
    b = load_best(tag)
    gain = 0
    for i in range(0, len(cs), step):
        t0 = time.time()
        try:
            p, log = tail_repair(cs[i], b[i], r0, tl, workers)
        except AssertionError as e:
            print(i, 'skip', str(e)[:60])
            continue
        g = len(b[i]) - len(p) if p is not None else 0
        gain += g
        print(f'#{i} len={len(b[i])} gain={g} {log} {time.time() - t0:.0f}s', flush=True)
    print('total gain', gain)


if __name__ == '__main__':
    main()
