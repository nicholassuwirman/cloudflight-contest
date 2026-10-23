"""Experiment: exact CP-SAT check for a steady-state copy cycle of period c (one acceptance per period).

usage: python exp_cpsat_cycle.py <tag> <idx> <c> [timelimit]
Free start config X (post-acceptance, empty board); c ticks; acceptance exactly on tick c; config(c) == X
(or X with an axis pair swapped). Skirt never receives matter (same hard prune as the beam).
"""
import sys
import time
from ortools.sat.python import cp_model
from l6core import parse_input
from l6model import PatternModel


def cycle_model(case, c, fix_start=None, cyclic=True, soft_skirt=False):
    pm = PatternModel(case)
    lm = pm.lm
    nl = lm.nl
    P, S = pm.P, pm.S
    m = cp_model.CpModel()
    L = [[m.NewIntVar(0, nl - 1, f'L{e}_{t}') for t in range(c + 1)] for e in range(6)]
    A = [[m.NewBoolVar(f'A{e}_{t}') for t in range(c + 1)] for e in range(6)]
    pos = [[[m.NewBoolVar('') for l in range(nl)] for t in range(c + 1)] for e in range(6)]
    for e in range(6):
        for t in range(c + 1):
            m.AddExactlyOne(pos[e][t])
            m.Add(L[e][t] == sum(l * pos[e][t][l] for l in range(nl)))
        for t in range(1, c + 1):
            up, dn, tg = m.NewBoolVar(''), m.NewBoolVar(''), m.NewBoolVar('')
            m.AddAtMostOne([up, dn, tg])
            m.Add(L[e][t] - L[e][t - 1] == up - dn)
            # A[t] = A[t-1] xor tg
            m.AddBoolXOr([A[e][t], A[e][t - 1], tg, True])
    if fix_start is not None:
        L0, A0 = fix_start
        for e in range(6):
            m.Add(L[e][0] == L0[e])
            m.Add(A[e][0] == A0[e])
    if cyclic:
        for ax in range(3):
            sw = m.NewBoolVar('')
            a, b = ax, ax + 3
            m.Add(L[a][c] == L[a][0]).OnlyEnforceIf(sw.Not())
            m.Add(L[b][c] == L[b][0]).OnlyEnforceIf(sw.Not())
            m.Add(A[a][c] == A[a][0]).OnlyEnforceIf(sw.Not())
            m.Add(A[b][c] == A[b][0]).OnlyEnforceIf(sw.Not())
            m.Add(L[a][c] == L[b][0]).OnlyEnforceIf(sw)
            m.Add(L[b][c] == L[a][0]).OnlyEnforceIf(sw)
            m.Add(A[a][c] == A[b][0]).OnlyEnforceIf(sw)
            m.Add(A[b][c] == A[a][0]).OnlyEnforceIf(sw)
    on = {}

    def beam(e, t, l):
        k = (e, t, l)
        if k not in on:
            v = m.NewBoolVar('')
            m.AddBoolAnd([pos[e][t][l], A[e][t]]).OnlyEnforceIf(v)
            m.AddBoolOr([pos[e][t][l].Not(), A[e][t].Not()]).OnlyEnforceIf(v.Not())
            on[k] = v
        return on[k]
    built = {}
    for qi, rc in enumerate(P + S):
        lines = lm.cell_lines[rc]
        isP = qi < len(P)
        prev = None
        for t in range(1, c + 1):
            hits = sum(beam(e, t, lines[e % 3]) for e in range(6))
            if not isP and not soft_skirt:
                m.Add(hits != 2)
                continue
            e2 = m.NewBoolVar('')
            g3 = m.NewBoolVar('')
            m.Add(hits == 2).OnlyEnforceIf(e2)
            m.Add(hits != 2).OnlyEnforceIf(e2.Not())
            m.Add(hits >= 3).OnlyEnforceIf(g3)
            m.Add(hits <= 2).OnlyEnforceIf(g3.Not())
            b = m.NewBoolVar('')
            if prev is None:
                m.Add(b == e2)
            else:
                # b = e2 or (prev and not g3)
                keep = m.NewBoolVar('')
                m.AddBoolAnd([prev, g3.Not()]).OnlyEnforceIf(keep)
                m.AddBoolOr([prev.Not(), g3]).OnlyEnforceIf(keep.Not())
                m.AddMaxEquality(b, [e2, keep])
            built[qi, t] = b
            prev = b
    nP = len(P)
    for q in range(nP):
        m.Add(built[q, c] == 1)
    if soft_skirt:
        for q in range(nP, nP + len(S)):
            m.Add(built[q, c] == 0)
    for t in range(1, c):
        lits = [built[q, t].Not() for q in range(nP)]
        if soft_skirt:
            lits += [built[q, t] for q in range(nP, nP + len(S))]
        m.AddBoolOr(lits)
    return m, L, A


def main():
    tag, idx, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 60
    case = parse_input(f'in_level-6_{tag}.txt')[idx]
    t0 = time.time()
    soft = len(sys.argv) > 5 and sys.argv[5] == 'soft'
    m, L, A = cycle_model(case, c, soft_skirt=soft)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = 8
    r = s.Solve(m)
    print(tag, idx, 'K', case['K'], 'c', c, 'soft' if soft else 'hard', s.StatusName(r), f'{time.time() - t0:.1f}s', flush=True)
    if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for t in range(c + 1):
            print('  ', [s.Value(L[e][t]) for e in range(6)], [s.Value(A[e][t]) for e in range(6)])


if __name__ == '__main__':
    main()
