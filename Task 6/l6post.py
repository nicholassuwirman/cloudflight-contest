"""Post-optimisation (improved-plan-1 §7): line deletion and adjacent-line merging.

Fast replay in line space with Python-int bitsets over pattern + skirt cells (skirt matter is tracked,
since an edited program may contaminate). Every kept edit is re-verified with simulate_ref.
"""
from l6core import simulate_ref
from l6model import PatternModel
from l6graph import program_to_actions
from l6solve import actions_to_program


class Replayer:
    def __init__(self, case, pm=None):
        pm = pm or PatternModel(case)
        self.pm = pm
        self.K = case['K']
        self.R = case['R']
        self.nl = pm.lm.nl

        def toint(arr):
            v = 0
            for w in range(len(arr)):
                v |= int(arr[w]) << (64 * w)
            return v
        self.lb = [[toint(pm.lineBits[a, l]) for l in range(self.nl)] for a in range(3)]
        self.Pm = toint(pm.Pm)
        self.Sm = toint(pm.Sm)

    def accepts(self, acts):
        """Tick (1-based) of the K-th acceptance, or None (also None on leaving the line window)."""
        L = [self.R] * 6
        A = [0] * 6
        M = 0
        n = 0
        lb, Pm, Sm, nl = self.lb, self.Pm, self.Sm, self.nl
        for t, act in enumerate(acts):
            for e in range(6):
                u = (act >> (2 * e)) & 3
                if u == 1:
                    L[e] += 1
                    if L[e] >= nl:
                        return None
                elif u == 2:
                    L[e] -= 1
                    if L[e] < 0:
                        return None
                elif u == 3:
                    A[e] ^= 1
            c0 = c1 = c2 = 0
            for e in range(6):
                if A[e]:
                    b = lb[e % 3][L[e]]
                    carry = c0 & b
                    c0 ^= b
                    c2 |= c1 & carry
                    c1 ^= carry
            e2 = c1 & ~c0 & ~c2
            g3 = (c1 & c0) | c2
            M = (M | e2) & ~g3
            if (M & Pm) == Pm and (M & Sm) == 0:
                M &= ~Pm
                n += 1
                if n == self.K:
                    return t + 1
        return None


def _merge(a, b):
    for e in range(6):
        if (a >> (2 * e)) & 3 and (b >> (2 * e)) & 3:
            return None
    return a | b


def post_optimize(case, prog, rounds=3):
    rp = Replayer(case)
    acts = program_to_actions(prog, rp.pm.sign)
    assert rp.accepts(acts) == len(acts)
    for _ in range(rounds):
        changed = False
        t = len(acts) - 1
        while t >= 0:
            cand = acts[:t] + acts[t + 1:]
            end = rp.accepts(cand)
            if end is not None:
                acts = cand[:end]
                changed = True
                t = min(t, len(acts)) - 1
                continue
            if t + 1 < len(acts):
                m = _merge(acts[t], acts[t + 1])
                if m is not None:
                    cand = acts[:t] + [m] + acts[t + 2:]
                    end = rp.accepts(cand)
                    if end is not None:
                        acts = cand[:end]
                        changed = True
                        t = min(t, len(acts)) - 1
                        continue
            t -= 1
        if not changed:
            break
    out = actions_to_program(acts, rp.pm.sign)
    acc = simulate_ref(case, out)
    if len(acc) == case['K'] and acc[-1] == len(out) and len(out) <= len(prog):
        return out
    return prog
