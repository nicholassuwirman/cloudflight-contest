"""Experiment: exact CP-SAT window repair (LNS). For a window [a, a+w) of the best program that contains no
acceptance, ask CP-SAT for a (w-1)-tick segment from the exact state at a to the exact config at a+w with
built(a+w) a superset of the original. On success splice, verify with simulate_ref, continue.

usage: python exp_window.py <tag> <idx> <w> [timelimit]
"""
import json
import sys
import time
from ortools.sat.python import cp_model
from l6core import parse_input, simulate_ref
from l6model import PatternModel
from l6graph import program_to_actions
from l6solve import actions_to_program


def trace(pm, acts, R):
    """Per tick t (state after t actions): (L, A, built set of P indices, acc)."""
    lm = pm.lm
    P = pm.P
    lines = [lm.cell_lines[rc] for rc in P]
    L, A = [R] * 6, [0] * 6
    built = set()
    acc = 0
    out = [(tuple(L), tuple(A), frozenset(built), acc)]
    for act in acts:
        for e in range(6):
            u = (act >> (2 * e)) & 3
            L[e] += 1 if u == 1 else (-1 if u == 2 else 0)
            A[e] ^= 1 if u == 3 else 0
        for q, ln in enumerate(lines):
            h = sum(1 for e in range(6) if A[e] and L[e] == ln[e % 3])
            if h == 2:
                built.add(q)
            elif h >= 3:
                built.discard(q)
        if len(built) == len(P):
            built = set()
            acc += 1
        out.append((tuple(L), tuple(A), frozenset(built), acc))
    return out


def window_model(pm, w, s0, s1):
    nl = pm.lm.nl
    P, S = pm.P, pm.S
    lm = pm.lm
    m = cp_model.CpModel()
    pos = [[[m.NewBoolVar('') for _ in range(nl)] for t in range(w + 1)] for e in range(6)]
    L = [[m.NewIntVar(0, nl - 1, '') for t in range(w + 1)] for e in range(6)]
    A = [[m.NewBoolVar('') for t in range(w + 1)] for e in range(6)]
    U = [[None] * (w + 1) for _ in range(6)]
    for e in range(6):
        for t in range(w + 1):
            m.AddExactlyOne(pos[e][t])
            m.Add(L[e][t] == sum(l * pos[e][t][l] for l in range(nl)))
        for t in range(1, w + 1):
            up, dn, tg = m.NewBoolVar(''), m.NewBoolVar(''), m.NewBoolVar('')
            m.AddAtMostOne([up, dn, tg])
            m.Add(L[e][t] - L[e][t - 1] == up - dn)
            m.AddBoolXOr([A[e][t], A[e][t - 1], tg, True])
            U[e][t] = (up, dn, tg)
        m.Add(L[e][0] == s0[0][e]); m.Add(A[e][0] == s0[1][e])
        m.Add(L[e][w] == s1[0][e]); m.Add(A[e][w] == s1[1][e])
    on = {}

    def beam(e, t, l):
        k = (e, t, l)
        if k not in on:
            v = m.NewBoolVar('')
            m.AddBoolAnd([pos[e][t][l], A[e][t]]).OnlyEnforceIf(v)
            m.AddBoolOr([pos[e][t][l].Not(), A[e][t].Not()]).OnlyEnforceIf(v.Not())
            on[k] = v
        return on[k]
    built_last = []
    for qi, rc in enumerate(P + S):
        ln = lm.cell_lines[rc]
        isP = qi < len(P)
        prev = None
        for t in range(1, w + 1):
            hits = sum(beam(e, t, ln[e % 3]) for e in range(6))
            if not isP:
                m.Add(hits != 2)
                continue
            e2, g3, b = m.NewBoolVar(''), m.NewBoolVar(''), m.NewBoolVar('')
            m.Add(hits == 2).OnlyEnforceIf(e2); m.Add(hits != 2).OnlyEnforceIf(e2.Not())
            m.Add(hits >= 3).OnlyEnforceIf(g3); m.Add(hits <= 2).OnlyEnforceIf(g3.Not())
            if prev is None:
                if qi in s0[2]:
                    m.AddMaxEquality(b, [e2, g3.Not()])
                else:
                    m.Add(b == e2)
            else:
                keep = m.NewBoolVar('')
                m.AddBoolAnd([prev, g3.Not()]).OnlyEnforceIf(keep)
                m.AddBoolOr([prev.Not(), g3]).OnlyEnforceIf(keep.Not())
                m.AddMaxEquality(b, [e2, keep])
            prev = b
            if t < w:
                built_last.append((qi, t, b))
        if isP and qi in s1[2]:
            m.Add(prev == 1)
    # no acceptance inside the window (keeps the acceptance schedule of the rest of the program)
    byt = {}
    for qi, t, b in built_last:
        byt.setdefault(t, []).append(b.Not())
    for t, lits in byt.items():
        if len(lits) == len(P):
            m.AddBoolOr(lits)
    return m, U


def main():
    tag, idx, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tl = float(sys.argv[4]) if len(sys.argv) > 4 else 20
    case = parse_input(f'in_level-6_{tag}.txt')[idx]
    prog = {int(k): v for k, v in json.load(open(f'best/{tag}.json')).items()}[idx]
    pm = PatternModel(case)
    acts = program_to_actions(prog, pm.sign)
    t0 = time.time()
    a = 0
    wins = tried = 0
    while a + w <= len(acts):
        tr = trace(pm, acts, case['R'])
        s0, s1 = tr[a], tr[a + w]
        if s0[3] != s1[3] or (a + w == len(acts)):
            a += 1
            continue
        tried += 1
        m, U = window_model(pm, w - 1, s0, s1)
        s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = tl; s.parameters.num_workers = 12
        r = s.Solve(m)
        if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            seg = []
            for t in range(1, w):
                act = 0
                for e in range(6):
                    up, dn, tg = U[e][t]
                    u = 1 if s.Value(up) else (2 if s.Value(dn) else (3 if s.Value(tg) else 0))
                    act |= u << (2 * e)
                seg.append(act)
            cand = acts[:a] + seg + acts[a + w:]
            p = actions_to_program(cand, pm.sign)
            acc = simulate_ref(case, p)
            if len(acc) == case['K'] and acc[-1] == len(p):
                acts = cand
                wins += 1
                print(f'  window at {a}: {w} -> {w - 1} ({s.StatusName(r)})', flush=True)
                continue
            print('  splice failed verification', acc, len(p))
        a += max(1, w // 2)
    print(tag, idx, 'old', len(prog), 'new', len(acts), 'windows tried', tried, 'wins', wins,
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
