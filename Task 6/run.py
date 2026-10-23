"""Parallel runner: solves every pattern of an input file, keeps the best verified program per pattern
in best/<file>.json, and writes out_level-6_<file>.txt.

usage: python run.py <file-tag> [W] [CAP] [extra key=value ...]
  file-tag: 0-example | 1-small | 2-medium | 3-large | 4-enormous
"""
import json
import os
import sys
import time
from multiprocessing import Pool

from l6core import parse_input, write_output, validate_file, simulate_ref

HERE = os.path.dirname(os.path.abspath(__file__))


def load_best(tag):
    path = os.path.join(HERE, 'best', f'{tag}.json')
    if os.path.exists(path):
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_best(tag, best):
    os.makedirs(os.path.join(HERE, 'best'), exist_ok=True)
    path = os.path.join(HERE, 'best', f'{tag}.json')
    with open(path + '.tmp', 'w') as f:
        json.dump({str(k): v for k, v in sorted(best.items())}, f)
    os.replace(path + '.tmp', path)


def work(args):
    i, case, params, inc = args
    params = dict(params)
    mode = params.pop('mode', 'beam')
    t0 = time.time()
    if mode == 'graph':
        from l6graph import solve_graph
        prog = solve_graph(case, incumbent=inc, **params)
    elif mode == 'sym':
        from l6sym import solve_sym
        prog = solve_sym(case, incumbent=inc, **params)
    elif mode == 'post':
        from l6post import post_optimize
        prog = post_optimize(case, inc)
    else:
        from l6solve import solve
        prog = solve(case, **params)
    return i, prog, time.time() - t0


def export(tag, cases, best):
    progs = [best[i] for i in range(len(cases))]
    out = os.path.join(HERE, f'out_level-6_{tag}.txt')
    write_output(out, progs)
    return out


def main():
    tag = sys.argv[1]
    params = {}
    if len(sys.argv) > 2:
        params['W'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        params['CAP'] = int(sys.argv[3])
    only = None
    for kv in sys.argv[4:]:
        k, v = kv.split('=')
        if k == 'only':
            only = set(int(x) for x in v.split(','))
            continue
        try:
            params[k] = float(v) if '.' in v else int(v)
        except ValueError:
            params[k] = v
    cases = parse_input(os.path.join(HERE, f'in_level-6_{tag}.txt'))
    best = load_best(tag)
    procs = int(params.pop('procs', os.cpu_count()))
    jobs = [(i, c, params, best.get(i)) for i, c in enumerate(cases) if only is None or i in only]
    # largest first for load balancing
    jobs.sort(key=lambda j: -(j[1]['R'] ** 2 * sum(r.count('M') for r in j[1]['grid'])))
    t0 = time.time()
    improved = 0
    with Pool(procs) as pool:
        for i, prog, dt in pool.imap_unordered(work, jobs):
            c = cases[i]
            if prog is None:
                print(f'[{i}] FAILED', flush=True)
                continue
            acc = simulate_ref(c, prog)
            assert len(acc) == c['K'] and acc[-1] == len(prog)
            old = len(best[i]) if i in best else None
            if old is None or len(prog) < old:
                best[i] = prog
                improved += 1
                save_best(tag, best)
            print(f'[{i}] R={c["R"]} K={c["K"]} lines={len(prog)} best={len(best[i])} (was {old}) {dt:.1f}s',
                  flush=True)
    if len(best) == len(cases):
        out = export(tag, cases, best)
        total, bad = validate_file(os.path.join(HERE, f'in_level-6_{tag}.txt'), out)
        print(f'{tag}: total={total} bad={bad} improved={improved} time={time.time() - t0:.0f}s')
    else:
        print(f'{tag}: {len(best)}/{len(cases)} solved')


if __name__ == '__main__':
    main()
