"""Certification / lower-bound pass with CP-SAT (improved-plan-2 section 6.1, 6.2).

Per pattern of best/<tag>.json:
  t1:   is the first build T1 optimal?  (T1-1 ticks from the start state infeasible)
  cmin: is the shortest repeat gap g a valid per-copy minimum?  (g-1 ticks from ANY start config with an
        empty board infeasible; valid because the relevant board is empty after every acceptance)
  lb = T1 (if certified, else 1) + (K-1) * cmin (if certified, else 1)
Hard-skirt model (the beam's rule). Results go to best/<tag>.cert.json; existing entries are kept unless the
incumbent changed.

usage: python cert.py <tag> [timelimit] [workers] [procs] [only=..]
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from l6core import parse_input, simulate_ref
from run import load_best, HERE


def check(args):
    i, case, prog, tl, workers = args
    from l6exact import Segment
    R = case['R']
    acc = simulate_ref(case, prog)
    gaps = [acc[0]] + [acc[j] - acc[j - 1] for j in range(1, len(acc))]
    T1 = gaps[0]
    res = dict(lines=len(prog), T1=T1, gaps=gaps)
    t0 = time.time()
    if T1 > 1:
        s = Segment(case, T1 - 1, start=([R] * 6, [0] * 6))
        res['t1'] = s.solve(tl=tl, workers=workers)
    else:
        res['t1'] = 'INFEASIBLE'
    if case['K'] > 1:
        g = min(gaps[1:])
        res['g'] = g
        if g > 1:
            s = Segment(case, g - 1)
            res['cmin'] = s.solve(tl=tl, workers=workers)
        else:
            res['cmin'] = 'INFEASIBLE'
    lb1 = T1 if res['t1'] == 'INFEASIBLE' else 1
    lbc = res.get('g', 1) if res.get('cmin') == 'INFEASIBLE' else 1
    res['lb'] = lb1 + (case['K'] - 1) * lbc
    res['time'] = time.time() - t0
    return i, res


def main():
    tag = sys.argv[1]
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    only = None
    for kv in sys.argv[5:]:
        k, v = kv.split('=')
        if k == 'only':
            only = set(int(x) for x in v.split(','))
    cases = parse_input(os.path.join(HERE, f'in_level-6_{tag}.txt'))
    best = load_best(tag)
    path = os.path.join(HERE, 'best', f'{tag}.cert.json')
    cert = {}
    if os.path.exists(path):
        cert = {int(k): v for k, v in json.load(open(path)).items()}
    jobs = []
    for i, c in enumerate(cases):
        if only is not None and i not in only:
            continue
        if i in cert and cert[i]['lines'] == len(best[i]) and 'UNKNOWN' not in (cert[i]['t1'], cert[i].get('cmin')):
            continue
        jobs.append((i, c, best[i], tl, workers))
    jobs.sort(key=lambda j: len(j[2]))
    with Pool(procs) as pool:
        for i, res in pool.imap_unordered(check, jobs):
            cert[i] = res
            with open(path + '.tmp', 'w') as f:
                json.dump({str(k): v for k, v in sorted(cert.items())}, f)
            os.replace(path + '.tmp', path)
            print(f'#{i} lines={res["lines"]} lb={res["lb"]} t1={res["t1"]} cmin={res.get("cmin")} '
                  f'gaps={res["gaps"][:6]} {res["time"]:.0f}s', flush=True)
    tot = sum(len(best[i]) for i in cert)
    lb = sum(v['lb'] for v in cert.values())
    print(f'{tag}: patterns={len(cert)} lines={tot} lb={lb} gap={tot - lb}')


if __name__ == '__main__':
    main()
