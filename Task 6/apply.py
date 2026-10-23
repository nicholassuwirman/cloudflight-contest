"""One pass of a deterministic operator over every pattern of a file; verified, monotone merge into best/.

usage: python apply.py <tag> <op> [procs]      op: pingpong | post
Do not run on a file while improve.py works on it (both write best/<tag>.json).
"""
import os
import sys
import time
from multiprocessing import Pool
from l6core import parse_input, simulate_ref, validate_file
from run import load_best, save_best, export, HERE


def work(args):
    i, case, prog, op = args
    if op == 'pingpong':
        from l6pp import solve_pingpong
        out = solve_pingpong(case, prog)
    else:
        from l6post import post_optimize
        out = post_optimize(case, prog)
    return i, out


def main():
    tag, op = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    cases = parse_input(os.path.join(HERE, f'in_level-6_{tag}.txt'))
    best = load_best(tag)
    t0 = time.time()
    before = sum(len(v) for v in best.values())
    jobs = [(i, c, best[i], op) for i, c in enumerate(cases)]
    with Pool(procs) as pool:
        for i, prog in pool.imap_unordered(work, jobs):
            if prog is None or len(prog) >= len(best[i]):
                continue
            acc = simulate_ref(cases[i], prog)
            if len(acc) == cases[i]['K'] and acc[-1] == len(prog):
                print(f'  #{i} {len(best[i])} -> {len(prog)}', flush=True)
                best[i] = prog
    save_best(tag, best)
    out = export(tag, cases, best)
    total, bad = validate_file(os.path.join(HERE, f'in_level-6_{tag}.txt'), out)
    print(f'{tag} {op}: {before} -> {total} bad={bad} {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
