"""Experiment: is a chord sweep realisable? Fix the coincident painter pair of axis `ax` to sweep the pattern's
chords one per tick (after moving from the centre and toggling on); CP-SAT chooses the other 4 sliders.
usage: python exp_sweep.py <tag> <idx> [extra_ticks] [timelimit] [direction +1/-1]"""
import sys, time, collections
from ortools.sat.python import cp_model
from l6core import parse_input
from l6model import PatternModel
from exp_cpsat_cycle import cycle_model
tag, idx = sys.argv[1], int(sys.argv[2])
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 300
dirn = int(sys.argv[5]) if len(sys.argv) > 5 else 1
case = parse_input(f'in_level-6_{tag}.txt')[idx]
pm = PatternModel(case); lm = pm.lm; R = case['R']
for ax in range(3):
    pl = collections.Counter(lm.cell_lines[rc][ax] for rc in pm.P)
    sk = collections.Counter(lm.cell_lines[rc][ax] for rc in pm.S)
    if all(sk[l] <= 4 for l in pl):
        break
lines = sorted(pl)
if dirn < 0:
    lines = lines[::-1]
first, last = lines[0], lines[-1]
span = abs(last - first) + 1
d = abs(first - R)
# painter schedule: ticks 1..d move both to `first`, tick d+1 toggle both on, then step one line per tick
sched = []
l = R
for t in range(d):
    l += 1 if first > R else -1
    sched.append((l, 0))
sched.append((l, 1))
for t in range(span - 1):
    l += dirn
    sched.append((l, 1))
T = len(sched) + extra
while len(sched) < T:
    sched.append(sched[-1])
print('axis', ax, 'chords', len(lines), 'span', span, 'setup', d + 1, 'horizon', T, flush=True)
t0 = time.time()
m, L, A = cycle_model(case, T, fix_start=([R] * 6, [0] * 6), cyclic=False)
for t, (lv, av) in enumerate(sched, start=1):
    for e in (ax, ax + 3):
        m.Add(L[e][t] == lv); m.Add(A[e][t] == av)
s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 12
r = s.Solve(m)
print(tag, idx, 'nP', pm.nP, 'sweep horizon', T, s.StatusName(r), f'{time.time() - t0:.0f}s', flush=True)
