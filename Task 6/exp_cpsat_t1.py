"""Experiment: exact CP-SAT check whether the first build can finish in c ticks from the start state.
usage: python exp_cpsat_t1.py <tag> <idx> <c> [timelimit]"""
import sys, time
from ortools.sat.python import cp_model
from l6core import parse_input
from exp_cpsat_cycle import cycle_model
tag, idx, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 60
case = parse_input(f'in_level-6_{tag}.txt')[idx]
R = case['R']
t0 = time.time()
m, L, A = cycle_model(case, c, fix_start=([R] * 6, [0] * 6), cyclic=False)
s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 12
r = s.Solve(m)
print(tag, idx, 'R', R, 'T1<=', c, s.StatusName(r), f'{time.time() - t0:.1f}s', flush=True)
