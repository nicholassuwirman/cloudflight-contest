"""CP-SAT copy-graph pass over the patterns that cert.py could not certify; verified, monotone merge.

usage: python exact_pass.py <tag> [budget] [workers] [procs] [only=..]
Do not run on a file while improve.py works on it (both write best/<tag>.json).
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from l6core import parse_input, simulate_ref, validate_file
from run import load_best, save_best, export, HERE


def work(args):
    i, case, prog, cmin, t1min, budget, workers = args
    from l6exactgraph import solve_exact_graph
    t0 = time.time()
    out = solve_exact_graph(case, prog, budget=budget, workers=workers, cmin=cmin, t1min=t1min)
    return i, out, time.time() - t0


def main():
    tag = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    only = None
    for kv in sys.argv[5:]:
        k, v = kv.split('=')
        if k == 'only':
            only = set(int(x) for x in v.split(','))
    cases = parse_input(os.path.join(HERE, f'in_level-6_{tag}.txt'))
    best = load_best(tag)
    cpath = os.path.join(HERE, 'best', f'{tag}.cert.json')
    cert = {int(k): v for k, v in json.load(open(cpath)).items()} if os.path.exists(cpath) else {}
    jobs = []
    for i, c in enumerate(cases):
        if only is not None and i not in only:
            continue
        v = cert.get(i)
        if v is not None and v['lb'] >= len(best[i]):
            continue
        cmin, t1min = 1, 1
        if v is not None and v['lines'] == len(best[i]):
            if v.get('cmin') == 'INFEASIBLE':
                cmin = v['g']
            if v['t1'] == 'INFEASIBLE':
                t1min = v['T1']
        jobs.append((i, c, best[i], cmin, t1min, budget, workers))
    jobs.sort(key=lambda j: -len(j[2]))
    print(f'{tag}: {len(jobs)} patterns', flush=True)
    before = sum(len(v) for v in best.values())
    t0 = time.time()
    with Pool(procs) as pool:
        for i, prog, dt in pool.imap_unordered(work, jobs):
            old = len(best[i])
            if prog is not None and len(prog) < old:
                acc = simulate_ref(cases[i], prog)
                if len(acc) == cases[i]['K'] and acc[-1] == len(prog):
                    best[i] = prog
                    save_best(tag, best)
            print(f'  #{i} {old} -> {len(best[i])} ({dt:.0f}s)', flush=True)
    out = export(tag, cases, best)
    total, bad = validate_file(os.path.join(HERE, f'in_level-6_{tag}.txt'), out)
    print(f'{tag} exact: {before} -> {total} bad={bad} {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
