"""Exhaustive search for steady-state cycles of period 1 and 2.

A period-2 cycle alternates configs X, Y (every slider does an action and then its reverse).
Per slider we only care about its beam: none or a line. Valid per-slider beam pairs (bX, bY):
equal, adjacent lines, or line <-> none (toggle). Lines missing pattern+skirt are equivalent to none.
"""
import numpy as np
from numba import njit
from l6beam import ZERO


def slider_options(rel_lines):
    """rel_lines: sorted list of relevant line indices (1..2R-1). -1 = none."""
    opts = set()
    vals = [-1] + list(rel_lines)
    for b in vals:
        opts.add((b, b))
        if b >= 0:
            opts.add((b, -1))
            opts.add((-1, b))
            if b + 1 in rel_lines:
                opts.add((b, b + 1))
                opts.add((b + 1, b))
    return sorted(opts)


def axis_table(pm, ax):
    """All distinct (oneX, twoX, oneY, twoY) for the slider pair of axis ax + a representative."""
    nl = pm.lm.nl
    rel = [l for l in range(1, nl - 1) if pm.lineBits[ax, l].any()]
    so = slider_options(rel)
    NW = pm.NW
    seen = {}
    for i in range(len(so)):
        for j in range(i, len(so)):
            (x1, y1), (x2, y2) = so[i], so[j]

            def bb(l):
                return pm.lineBits[ax, l] if l >= 0 else np.zeros(NW, dtype=np.uint64)
            bx1, bx2, by1, by2 = bb(x1), bb(x2), bb(y1), bb(y2)
            key = np.concatenate([bx1 ^ bx2, bx1 & bx2, by1 ^ by2, by1 & by2])
            k = key.tobytes()
            if k not in seen:
                seen[k] = (key, (x1, y1, x2, y2))
    keys = np.array([v[0] for v in seen.values()], dtype=np.uint64).reshape(len(seen), 4, NW)
    reps = np.array([v[1] for v in seen.values()], dtype=np.int64)
    return keys, reps


@njit(cache=True)
def _e2g3(o0, t0, o1, t1, o2, t2):
    a0 = o0 | t0; a1 = o1 | t1; a2 = o2 | t2
    e2 = ((t0 & ~a1 & ~a2) | (t1 & ~a0 & ~a2) | (t2 & ~a0 & ~a1) |
          (o0 & o1 & ~a2) | (o0 & o2 & ~a1) | (o1 & o2 & ~a0))
    g3 = (t0 & (a1 | a2)) | (t1 & (a0 | a2)) | (t2 & (a0 | a1)) | (o0 & o1 & o2)
    return e2, g3


@njit(cache=True)
def enum_cycles(tH, tD, tE, Pm, Sm, NW, maxOut):
    """Returns list of (iH, iD, iE, kind): kind 1 = accepts every tick, 2 = accepts at Y every 2 ticks,
    3 = accepts at X every 2 ticks."""
    out = np.zeros((maxOut, 4), dtype=np.int64)
    n = 0
    for iH in range(tH.shape[0]):
        for iD in range(tD.shape[0]):
            for iE in range(tE.shape[0]):
                okX = True; okY = True
                fullX = True; fullY = True; accY = True; accX = True
                for w in range(NW):
                    eX, gX = _e2g3(tH[iH, 0, w], tH[iH, 1, w], tD[iD, 0, w], tD[iD, 1, w], tE[iE, 0, w], tE[iE, 1, w])
                    eY, gY = _e2g3(tH[iH, 2, w], tH[iH, 3, w], tD[iD, 2, w], tD[iD, 3, w], tE[iE, 2, w], tE[iE, 3, w])
                    if (eX & Sm[w]) != ZERO:
                        okX = False
                    if (eY & Sm[w]) != ZERO:
                        okY = False
                    if not (okX and okY):
                        break
                    p = Pm[w]
                    if (eX & p) != p:
                        fullX = False
                    if (eY & p) != p:
                        fullY = False
                    if (((eX & ~gY) | eY) & p) != p:
                        accY = False
                    if (((eY & ~gX) | eX) & p) != p:
                        accX = False
                if not (okX and okY):
                    continue
                kind = 0
                if fullX and fullY:
                    kind = 1
                elif accY:
                    kind = 2
                elif accX:
                    kind = 3
                if kind and n < maxOut:
                    out[n, 0] = iH; out[n, 1] = iD; out[n, 2] = iE; out[n, 3] = kind
                    n += 1
    return out[:n]


def find_cycles(pm, maxOut=200000):
    tabs = [axis_table(pm, a) for a in range(3)]
    res = enum_cycles(tabs[0][0], tabs[1][0], tabs[2][0], pm.Pm, pm.Sm, pm.NW, maxOut)
    cyc = []
    for iH, iD, iE, kind in res:
        reps = [tabs[0][1][iH], tabs[1][1][iD], tabs[2][1][iE]]
        cyc.append((int(kind), reps))
    return cyc, [len(t[0]) for t in tabs]
