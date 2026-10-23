"""CP-SAT model of c ticks of the machine (improved-plan-2 section 6): oracle, lower bounds, exact segments.

segment_model(case, c, start, built0, ...) models ticks 1..c from a start config (fixed, or free when start is
None) and a board whose relevant matter is built0 (set of P indices; the skirt is clean). The last tick holds the
next acceptance and no earlier tick accepts. Skirt hard (never receives matter, like the beam) or soft (may be
contaminated, must be clean at the acceptance and at every tick where P is full).
Actions are read back from the move variables, so a FEASIBLE answer is a concrete segment.
"""
import time
from ortools.sat.python import cp_model
from l6model import PatternModel

STATUS = {cp_model.OPTIMAL: 'FEASIBLE', cp_model.FEASIBLE: 'FEASIBLE', cp_model.INFEASIBLE: 'INFEASIBLE',
          cp_model.UNKNOWN: 'UNKNOWN', cp_model.MODEL_INVALID: 'INVALID'}


class Segment:
    def __init__(self, case, c, start=None, built0=(), soft=False, end=None, cyclic=False, pm=None,
                 accept=True):
        pm = pm or PatternModel(case)
        self.pm, self.c = pm, c
        lm = pm.lm
        nl = lm.nl
        P, S = pm.P, pm.S
        m = cp_model.CpModel()
        self.m = m
        pos = [[[m.NewBoolVar('') for _ in range(nl)] for t in range(c + 1)] for e in range(6)]
        L = [[m.NewIntVar(0, nl - 1, '') for t in range(c + 1)] for e in range(6)]
        A = [[m.NewBoolVar('') for t in range(c + 1)] for e in range(6)]
        U = [[None] * (c + 1) for _ in range(6)]
        for e in range(6):
            for t in range(c + 1):
                m.AddExactlyOne(pos[e][t])
                m.Add(L[e][t] == sum(l * pos[e][t][l] for l in range(nl)))
            for t in range(1, c + 1):
                up, dn, tg = m.NewBoolVar(''), m.NewBoolVar(''), m.NewBoolVar('')
                m.AddAtMostOne([up, dn, tg])
                m.Add(L[e][t] - L[e][t - 1] == up - dn)
                m.AddBoolXOr([A[e][t], A[e][t - 1], tg, True])
                U[e][t] = (up, dn, tg)
        self.L, self.A, self.U, self.pos = L, A, U, pos
        if start is not None:
            L0, A0 = start
            for e in range(6):
                m.Add(L[e][0] == int(L0[e]))
                m.Add(A[e][0] == int(A0[e]))
        if end is not None:
            L1, A1 = end
            for e in range(6):
                m.Add(L[e][c] == int(L1[e]))
                m.Add(A[e][c] == int(A1[e]))
        if cyclic:
            for ax in range(3):
                sw = m.NewBoolVar('')
                a, b = ax, ax + 3
                for x, y in ((L, L), (A, A)):
                    m.Add(x[a][c] == y[a][0]).OnlyEnforceIf(sw.Not())
                    m.Add(x[b][c] == y[b][0]).OnlyEnforceIf(sw.Not())
                    m.Add(x[a][c] == y[b][0]).OnlyEnforceIf(sw)
                    m.Add(x[b][c] == y[a][0]).OnlyEnforceIf(sw)
        on = {}

        def beam(e, t, l):
            k = (e, t, l)
            if k not in on:
                v = m.NewBoolVar('')
                m.AddBoolAnd([pos[e][t][l], A[e][t]]).OnlyEnforceIf(v)
                m.AddBoolOr([pos[e][t][l].Not(), A[e][t].Not()]).OnlyEnforceIf(v.Not())
                on[k] = v
            return on[k]
        built0 = set(built0)
        built = {}
        nP = len(P)
        for qi, rc in enumerate(P + S):
            lines = lm.cell_lines[rc]
            isP = qi < nP
            prev = None
            for t in range(1, c + 1):
                hits = sum(beam(e, t, lines[e % 3]) for e in range(6))
                if not isP and not soft:
                    m.Add(hits != 2)
                    continue
                e2 = m.NewBoolVar('')
                g3 = m.NewBoolVar('')
                m.Add(hits == 2).OnlyEnforceIf(e2)
                m.Add(hits != 2).OnlyEnforceIf(e2.Not())
                m.Add(hits >= 3).OnlyEnforceIf(g3)
                m.Add(hits <= 2).OnlyEnforceIf(g3.Not())
                b = m.NewBoolVar('')
                if prev is None and qi not in built0:
                    m.Add(b == e2)
                else:
                    keep = m.NewBoolVar('')
                    if prev is None:
                        m.Add(keep == g3.Not())
                    else:
                        m.AddBoolAnd([prev, g3.Not()]).OnlyEnforceIf(keep)
                        m.AddBoolOr([prev.Not(), g3]).OnlyEnforceIf(keep.Not())
                    m.AddMaxEquality(b, [e2, keep])
                built[qi, t] = b
                prev = b
        self.built = built
        S0 = range(nP, nP + len(S))
        if accept:
            for q in range(nP):
                m.Add(built[q, c] == 1)
            if soft:
                for q in S0:
                    m.Add(built[q, c] == 0)
        # no acceptance before tick c (an acceptance needs P full and a clean skirt)
        for t in range(1, c if accept else c + 1):
            lits = [built[q, t].Not() for q in range(nP)]
            if soft:
                lits += [built[q, t] for q in S0]
            m.AddBoolOr(lits)

    def hint(self, acts, start):
        """Hint a concrete action sequence (len c) from a fixed start config."""
        L, A = list(start[0]), list(start[1])
        for e in range(6):
            self.m.AddHint(self.L[e][0], L[e])
            self.m.AddHint(self.A[e][0], A[e])
        for t, act in enumerate(acts, start=1):
            for e in range(6):
                u = (act >> (2 * e)) & 3
                up, dn, tg = self.U[e][t]
                self.m.AddHint(up, int(u == 1))
                self.m.AddHint(dn, int(u == 2))
                self.m.AddHint(tg, int(u == 3))
                L[e] += 1 if u == 1 else (-1 if u == 2 else 0)
                A[e] ^= 1 if u == 3 else 0
                self.m.AddHint(self.L[e][t], L[e])
                self.m.AddHint(self.A[e][t], A[e])

    def solve(self, tl=60.0, workers=8, log=False):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = tl
        s.parameters.num_workers = workers
        s.parameters.log_search_progress = log
        t0 = time.time()
        r = s.Solve(self.m)
        self.solver = s
        self.status = STATUS.get(r, str(r))
        self.time = time.time() - t0
        return self.status

    def actions(self):
        s = self.solver
        acts = []
        for t in range(1, self.c + 1):
            act = 0
            for e in range(6):
                up, dn, tg = self.U[e][t]
                u = 1 if s.Value(up) else (2 if s.Value(dn) else (3 if s.Value(tg) else 0))
                act |= u << (2 * e)
            acts.append(act)
        return acts

    def config(self, t):
        s = self.solver
        return (tuple(s.Value(self.L[e][t]) for e in range(6)), tuple(s.Value(self.A[e][t]) for e in range(6)))
