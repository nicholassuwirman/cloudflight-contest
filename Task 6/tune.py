"""Parameter tuning on a fixed benchmark subset (does not touch best/).

usage: python tune.py "<json list of param dicts>" [bench]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

from l6core import parse_input

HERE = os.path.dirname(os.path.abspath(__file__))


def bench_cases(name):
    out = []
    if name in ('mixed', 'medium'):
        cs = parse_input(os.path.join(HERE, 'in_level-6_2-medium.txt'))
        out += [('2-medium', i, cs[i]) for i in range(0, 210, 7)]
    if name in ('mixed', 'large'):
        cs = parse_input(os.path.join(HERE, 'in_level-6_3-large.txt'))
        out += [('3-large', i, cs[i]) for i in range(0, 190, 12)]
    if name == 'enormous':
        cs = parse_input(os.path.join(HERE, 'in_level-6_4-enormous.txt'))
        out += [('4-enormous', i, cs[i]) for i in range(0, 140, 14)]
    return out


def work(args):
    k, tag, i, case, params = args
    from l6solve import solve
    t0 = time.time()
    prog = solve(case, **params)
    return k, tag, i, (len(prog) if prog else 10 ** 6), time.time() - t0


def main():
    plist = json.loads(sys.argv[1])
    bench = sys.argv[2] if len(sys.argv) > 2 else 'mixed'
    cases = bench_cases(bench)
    jobs = [(k, tag, i, c, p) for k, p in enumerate(plist) for (tag, i, c) in cases]
    jobs.sort(key=lambda j: -(j[3]['R'] ** 2 * sum(r.count('M') for r in j[3]['grid'])))
    res = {}
    with Pool(os.cpu_count()) as pool:
        for k, tag, i, n, dt in pool.imap_unordered(work, jobs):
            r = res.setdefault(k, {'tot': {}, 'time': 0.0})
            r['tot'][tag] = r['tot'].get(tag, 0) + n
            r['time'] += dt
    for k, p in enumerate(plist):
        r = res[k]
        print(json.dumps(p), r['tot'], 'sum', sum(r['tot'].values()), f"cpu {r['time']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
