"""Anytime improvement driver (improved-plan-2 §4).

Runs operators on the incumbent programs of one input file for a time budget and merges every verified
improvement into best/<tag>.json immediately (monotone). The out file is rewritten periodically, at the end
and on Ctrl-C, so an interruption never leaves it stale.

usage: python improve.py <tag> [minutes] [k=v ...]
  procs=N          worker processes (default: all cores)
  ops=a:w,b:w      operators and initial weights (sym, suffix, graph, post)
  only=1,2,3       restrict to these pattern indices
  Wsym=, Wsuf=     beam widths (defaults per file)
  seed=S           base seed (job j uses seed S + j)
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from l6core import parse_input, simulate_ref, validate_file
from run import load_best, save_best, export, HERE

# --------------------------------------------------------------------------------------------------------------
# worker side


def _verify(case, prog):
    acc = simulate_ref(case, prog)
    return len(acc) == case['K'] and acc[-1] == len(prog)


def state_at(pm, acts, t, R):
    """Exact state after t actions: (L, A, built ∩ P as uint64 words, acc). Asserts the skirt stays clean."""
    nl = pm.lm.nl
    lb = [[sum(int(pm.lineBits[a, l, w]) << (64 * w) for w in range(pm.NW)) for l in range(nl)] for a in range(3)]
    Pm = sum(int(pm.Pm[w]) << (64 * w) for w in range(pm.NW))
    Sm = sum(int(pm.Sm[w]) << (64 * w) for w in range(pm.NW))
    L = [R] * 6
    A = [0] * 6
    M = 0
    acc = 0
    for act in acts[:t]:
        for e in range(6):
            u = (act >> (2 * e)) & 3
            if u == 1:
                L[e] += 1
            elif u == 2:
                L[e] -= 1
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
        M = (M | (c1 & ~c0 & ~c2)) & ~((c1 & c0) | c2)
        assert not (M & Sm)
        if (M & Pm) == Pm:
            M &= ~Pm
            acc += 1
    M &= Pm
    built = np.array([(M >> (64 * w)) & 0xFFFFFFFFFFFFFFFF for w in range(pm.NW)], dtype=np.uint64)
    return np.array(L, dtype=np.int64), np.array(A, dtype=np.int64), built, acc


def map_fwd(prog, perm, m):
    """Inverse of l6sym.map_back: original-frame program -> transformed frame."""
    from l6sym import FLIP
    return [[FLIP[line[perm[j]]] if m else line[perm[j]] for j in range(6)] for line in prog]


def pick_cut(case, prog, rng):
    """Cut tick biased towards long copy segments (a segment is chosen with probability ∝ its length)."""
    acc = simulate_ref(case, prog)
    segs = []
    prev = 0
    for t in acc:
        segs.append((prev, t))
        prev = t
    tot = sum(b - a for a, b in segs)
    x = rng.random() * tot
    for a, b in segs:
        if x < b - a:
            return a + min(b - a - 1, int(x))
        x -= b - a
    return segs[-1][0]


def run_beam(pm, K, L0, A0, b0, acc0, W, maxT, par, seed):
    """beam_search, or beam_search2 when the parameter set uses its extra features (wC, wL)."""
    if par.get('wC', 0) or par.get('wL', 0):
        from l6beam2 import beam_search2
        W = max(50, int(W * par.get('wf', 1.0)))
        return beam_search2(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, K,
                            L0, A0, b0, acc0, W, 10 * W, maxT, par['iB'], par['iP'], par['wS'], par['wM'], seed, 0,
                            float(par.get('wL', 0.0)), int(par.get('LX', 0)), -1, 0, float(par.get('wC', 0.0)))
    from l6beam import beam_search
    return beam_search(pm.lineBits, pm.Pm, pm.Sm, pm.NW, pm.nP, pm.cellLine, pm.lm.nl, K,
                       L0, A0, b0, acc0, W, 10 * W, maxT, par['iB'], par['iP'], par['wS'], par['wM'], seed, 0)


def op_sym(case, inc, p):
    from l6sym import transform_case, sym_info, map_back
    from l6model import PatternModel
    from l6solve import actions_to_program
    k, m = p['sym']
    q = transform_case(case, k, m)
    pm = PatternModel(q)
    R = q['R']
    d, path = run_beam(pm, q['K'], np.full(6, R, dtype=np.int64), np.zeros(6, dtype=np.int64),
                       np.zeros(pm.NW, dtype=np.uint64), 0, p['W'], len(inc) - 1, p['par'], p['seed'])
    if d < 0:
        return None
    _, perm = sym_info(R, k, m)
    return map_back(actions_to_program(path, pm.sign), perm, m)


def op_suffix(case, inc, p):
    from l6sym import transform_case, sym_info, map_back
    from l6model import PatternModel
    from l6graph import program_to_actions
    from l6solve import actions_to_program
    rng = random.Random(p['seed'])
    k, m = p['sym']
    R = case['R']
    q = transform_case(case, k, m) if (k, m) != (0, 0) else case
    _, perm = sym_info(R, k, m)
    pm = PatternModel(q)
    acts = program_to_actions(map_fwd(inc, perm, m), pm.sign)
    best = list(acts)
    W = p['W']
    par = p['par']
    for _ in range(p['cuts']):
        cur = map_back(actions_to_program(best, pm.sign), perm, m)
        t = pick_cut(case, cur, rng)
        L0, A0, b0, acc0 = state_at(pm, best, t, R)
        maxT = len(best) - t - 1
        if maxT <= 0:
            continue
        d, path = run_beam(pm, case['K'], L0, A0, b0, acc0, W, maxT, par, rng.randrange(1 << 30))
        if d >= 0 and t + d < len(best):
            cand = best[:t] + [int(x) for x in path]
            if _verify(q, actions_to_program(cand, pm.sign)):
                best = cand
    if len(best) >= len(inc):
        return None
    return map_back(actions_to_program(best, pm.sign), perm, m)


def op_graph(case, inc, p):
    from l6graph import solve_graph
    par = p['par']
    return solve_graph(case, incumbent=inc, budget=p['budget'], W=p['W'], CAP=10 * p['W'], slack=p['slack'],
                       seed=p['seed'], iB=par['iB'], iP=par['iP'], wS=par['wS'], wM=par['wM'], pp=True)


def op_post(case, inc, p):
    from l6post import post_optimize
    return post_optimize(case, inc)


def op_pingpong(case, inc, p):
    from l6pp import solve_pingpong
    return solve_pingpong(case, inc)


def polish(case, prog):
    """Cheap deterministic finishers applied to every improvement: ping-pong edges (K>=2), line deletion."""
    from l6post import post_optimize
    if case['K'] >= 2:
        from l6pp import solve_pingpong
        pp = solve_pingpong(case, prog)
        if pp is not None and len(pp) < len(prog):
            prog = pp
    return post_optimize(case, prog)


OPS = {'sym': op_sym, 'suffix': op_suffix, 'graph': op_graph, 'post': op_post, 'pingpong': op_pingpong}


def work(job):
    jid, op, i, case, inc, p = job
    pj = p.get('pj')
    t0 = time.time()
    try:
        prog = OPS[op](case, inc, p)
    except AssertionError:
        prog = None
    if prog is not None and len(prog) < len(inc) and _verify(case, prog):
        if op != 'post':
            prog = polish(case, prog)
    else:
        prog = None
    return jid, op, i, prog, time.time() - t0, pj


# --------------------------------------------------------------------------------------------------------------
# driver side

PORTFOLIO = [
    dict(iB=4, iP=1, wS=0.05, wM=0.5),
    dict(iB=2, iP=1, wS=0.05, wM=2.0),
    dict(iB=4, iP=0, wS=0.2, wM=2.0),
    dict(iB=4, iP=1, wS=0.0, wM=0.5),
    # l6beam2 variants: missing-chord-count penalty (wC), 1-ply lookahead (wL, ~4x cost -> narrower beam wf)
    dict(iB=4, iP=1, wS=0.05, wM=0.5, wC=2.0),
    dict(iB=4, iP=1, wS=0.05, wM=0.5, wC=6.0),
    dict(iB=2, iP=1, wS=0.05, wM=2.0, wC=3.0),
    dict(iB=4, iP=1, wS=0.05, wM=0.5, wL=0.3, LX=0, wf=0.25),
    dict(iB=4, iP=1, wS=0.05, wM=0.5, wL=0.5, LX=0, wf=0.25, wC=2.0),
]
SYMS = [(k, m) for m in (0, 1) for k in range(6)]

DEFAULTS = {
    '0-example': dict(Wsym=2000, Wsuf=2000, ops='sym:1,suffix:1'),
    '1-small': dict(Wsym=3000, Wsuf=3000, ops='sym:1,suffix:1,graph:1'),
    '2-medium': dict(Wsym=1000, Wsuf=1500, ops='sym:1,suffix:1,graph:2', gW=1000, gbudget=30),
    '3-large': dict(Wsym=1600, Wsuf=3200, ops='sym:1,suffix:2,graph:1', gW=600, gbudget=60),
    '4-enormous': dict(Wsym=1600, Wsuf=3200, ops='sym:1,suffix:2,graph:1', gW=800, gbudget=90),
}


class Driver:
    def __init__(self, tag, opts):
        self.tag = tag
        self.cases = parse_input(os.path.join(HERE, f'in_level-6_{tag}.txt'))
        self.best = load_best(tag)
        assert len(self.best) == len(self.cases), 'run run.py first so every pattern has an incumbent'
        self.opts = dict(DEFAULTS[tag])
        self.opts.update(opts)
        self.ops = {}
        for kv in str(self.opts['ops']).split(','):
            name, w = kv.split(':')
            self.ops[name] = float(w)
        only = self.opts.get('only')
        self.idx = [int(x) for x in str(only).split(',')] if only else list(range(len(self.cases)))
        cpath = os.path.join(HERE, 'best', f'{tag}.cert.json')
        if os.path.exists(cpath):
            # skip patterns whose incumbent equals the certified lower bound (cert.py)
            cert = {int(k): v for k, v in json.load(open(cpath)).items()}
            done = {i for i, v in cert.items() if v['lb'] >= len(self.best[i])}
            self.idx = [i for i in self.idx if i not in done]
            print(f'{tag}: {len(done)} patterns certified optimal, skipped', flush=True)
        self.fails = {i: 0 for i in self.idx}
        self.inflight = {i: 0 for i in self.idx}
        self.stat = {op: [0, 0, 0, 0.0] for op in self.ops}     # tries, wins, saved, cpu
        npf = int(self.opts.get('npf', len(PORTFOLIO)))
        self.pstat = {j: [0, 0, 0, 0.0] for j in range(npf)}
        self.seed = int(self.opts.get('seed', 1000))
        self.rng = random.Random(self.seed)
        self.jid = 0
        self.dirty = 0
        self.last_export = time.time()

    def total(self):
        return sum(len(v) for v in self.best.values())

    def op_weight(self, op):
        tries, wins, saved, cpu = self.stat[op]
        base = self.ops[op]
        if tries < 8:
            return base
        return base * (0.25 + (saved + 1.0) / (cpu / 60.0 + 4.0))

    def pf_weight(self, j):
        tries, wins, saved, cpu = self.pstat[j]
        if tries < 6:
            return 1.0
        return 0.25 + (saved + 1.0) / (cpu / 60.0 + 4.0)

    def pick_pattern(self, op):
        cand = [i for i in self.idx if (op not in ('graph', 'pingpong') or self.cases[i]['K'] >= 2)]
        if not cand:
            return None
        ws = [len(self.best[i]) * (0.8 ** self.fails[i]) / (1 + 3 * self.inflight[i]) for i in cand]
        return self.rng.choices(cand, weights=ws)[0]

    def next_job(self):
        ops = list(self.ops)
        for _ in range(20):
            op = self.rng.choices(ops, weights=[self.op_weight(o) for o in ops])[0]
            i = self.pick_pattern(op)
            if i is not None:
                break
        self.jid += 1
        pj = self.rng.choices(list(self.pstat), weights=[self.pf_weight(j) for j in self.pstat])[0]
        p = dict(seed=self.seed + self.jid, par=PORTFOLIO[pj], pj=pj)
        if op == 'sym':
            p.update(sym=self.rng.choice(SYMS), W=int(self.opts['Wsym']))
        elif op == 'suffix':
            p.update(sym=self.rng.choice(SYMS), W=int(self.opts['Wsuf']), cuts=int(self.opts.get('cuts', 3)))
        elif op == 'graph':
            p.update(W=int(self.opts.get('gW', 600)), budget=float(self.opts.get('gbudget', 60)),
                     slack=int(self.opts.get('slack', 2)))
        self.inflight[i] += 1
        return (self.jid, op, i, self.cases[i], self.best[i], p)

    def merge(self, op, i, prog, dt, pj=None):
        st = self.stat[op]
        st[0] += 1
        st[3] += dt
        self.inflight[i] -= 1
        ps = self.pstat.get(pj) if op in ('sym', 'suffix') else None
        if ps is not None:
            ps[0] += 1
            ps[3] += dt
        if prog is None:
            self.fails[i] += 1
            return
        c = self.cases[i]
        if not _verify(c, prog) or len(prog) >= len(self.best[i]):
            self.fails[i] += 1
            return
        old = len(self.best[i])
        self.best[i] = prog
        save_best(self.tag, self.best)
        st[1] += 1
        st[2] += old - len(prog)
        if ps is not None:
            ps[1] += 1
            ps[2] += old - len(prog)
        self.fails[i] = 0
        self.dirty += 1
        print(f'  [{op}] #{i} {old} -> {len(prog)}  total={self.total()}  ({dt:.0f}s)', flush=True)

    def maybe_export(self, force=False):
        if self.dirty and (force or self.dirty >= 10 or time.time() - self.last_export > 300):
            export(self.tag, self.cases, self.best)
            self.dirty = 0
            self.last_export = time.time()

    def report(self, t0):
        s = ' '.join(f'{op}:{v[0]}/{v[1]}/-{v[2]}/{v[3]:.0f}s' for op, v in self.stat.items())
        print(f'[{(time.time() - t0) / 60:.1f} min] total={self.total()} jobs={self.jid}  {s}', flush=True)
        print('   portfolio ' + ' '.join(f'{j}:{v[0]}/{v[1]}/-{v[2]}/{v[3]:.0f}s' for j, v in self.pstat.items()),
              flush=True)

    def run(self, minutes, procs):
        t0 = time.time()
        deadline = t0 + 60 * minutes
        print(f'{self.tag}: start total={self.total()} ops={self.ops} procs={procs}', flush=True)
        last_rep = t0
        pending = set()
        ex = ProcessPoolExecutor(procs)
        try:
            while time.time() < deadline or pending:
                while time.time() < deadline and len(pending) < procs:
                    pending.add(ex.submit(work, self.next_job()))
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for fu in done:
                    r = fu.result()
                    self.merge(r[1], r[2], r[3], r[4], r[5] if len(r) > 5 else None)
                self.maybe_export()
                if time.time() - last_rep > 120:
                    self.report(t0)
                    last_rep = time.time()
        except KeyboardInterrupt:
            print('interrupted', flush=True)
            for fu in pending:
                fu.cancel()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            self.maybe_export(force=True)
            export(self.tag, self.cases, self.best)
            self.report(t0)
            tot, bad = validate_file(os.path.join(HERE, f'in_level-6_{self.tag}.txt'),
                                     os.path.join(HERE, f'out_level-6_{self.tag}.txt'), verbose=True)
            print(f'{self.tag}: final total={tot} bad={bad}', flush=True)


def main():
    tag = sys.argv[1]
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    opts = {}
    for kv in sys.argv[3:]:
        k, v = kv.split('=', 1)
        opts[k] = v
    procs = int(opts.pop('procs', os.cpu_count()))
    Driver(tag, opts).run(minutes, procs)


if __name__ == '__main__':
    main()
