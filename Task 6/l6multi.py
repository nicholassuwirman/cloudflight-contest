"""Multi-endpoint single-copy beam (edge generator for the copy-graph DP).

Same layered beam as l6beam.beam_search, but starting from an arbitrary config with an empty board and
target one acceptance. Accepting children are recorded as endpoints (deduplicated by canonical config) and
kept out of the beam; the search continues `slack` layers past the first acceptance.
"""
import numpy as np
from numba import njit
from l6beam import (ZERO, popcnt, axis_options, apply_action, stamp, canon_hash, penalty)


@njit(cache=True)
def config_hash(L, A):
    h = np.uint64(0x12345)
    for ax in range(3):
        c1 = L[ax] * 2 + A[ax]
        c2 = L[ax + 3] * 2 + A[ax + 3]
        code = c1 * 64 + c2 if c1 <= c2 else c2 * 64 + c1
        h = h * np.uint64(4099) + np.uint64(code + 1)
    return h


@njit(cache=True)
def beam_multi(lineBits, Pm, Sm, NW, nP, cellLine, nl,
               L0, A0, W, CAP, maxT, slack, MAXE,
               iB, iP, wS, wM, seed, jit):
    """Returns (nEnd, depth[nEnd], endL[nEnd,6], endA[nEnd,6], paths[nEnd, maxT])."""
    np.random.seed(seed)
    bL = np.zeros((W, 6), dtype=np.int64)
    bA = np.zeros((W, 6), dtype=np.int64)
    bB = np.zeros((W, NW), dtype=np.uint64)
    nb = 1
    for e in range(6):
        bL[0, e] = L0[e]
        bA[0, e] = A0[e]
    parArr = np.zeros((maxT, W), dtype=np.int32)
    actArr = np.zeros((maxT, W), dtype=np.int32)

    eDepth = np.zeros(MAXE, dtype=np.int64)
    eL = np.zeros((MAXE, 6), dtype=np.int64)
    eA = np.zeros((MAXE, 6), dtype=np.int64)
    ePath = np.zeros((MAXE, maxT), dtype=np.int32)
    eHash = np.zeros(MAXE, dtype=np.uint64)
    nEnd = 0
    firstD = -1

    KB = iB * (nP + 1) + iP * (nP + 1) + 1
    maxKey = KB + 1
    hist = np.zeros(maxKey + 1, dtype=np.int64)
    CAPS = max(CAP * 4, 1 << 16)
    cPar = np.zeros(CAPS, dtype=np.int32)
    cAct = np.zeros(CAPS, dtype=np.int32)
    cKey = np.zeros(CAPS, dtype=np.int64)

    optAct = np.zeros((3, 16), dtype=np.int64)
    optOne = np.zeros((3, 16, NW), dtype=np.uint64)
    optTwo = np.zeros((3, 16, NW), dtype=np.uint64)
    optCode = np.zeros((3, 16), dtype=np.int64)
    nbw = np.zeros(NW, dtype=np.uint64)

    nL = np.zeros(6, dtype=np.int64)
    nA = np.zeros(6, dtype=np.int64)
    e2 = np.zeros(NW, dtype=np.uint64)
    g3 = np.zeros(NW, dtype=np.uint64)
    tau1 = np.zeros((3, nl), dtype=np.int64)
    tau2 = np.zeros((3, nl), dtype=np.int64)

    MAXC = 2 * CAP
    sL = np.zeros((MAXC, 6), dtype=np.int64)
    sA = np.zeros((MAXC, 6), dtype=np.int64)
    sB = np.zeros((MAXC, NW), dtype=np.uint64)
    sPar = np.zeros(MAXC, dtype=np.int32)
    sAct = np.zeros(MAXC, dtype=np.int32)
    sScore = np.zeros(MAXC, dtype=np.float64)
    HT = 1
    while HT < 4 * MAXC:
        HT *= 2
    htab = np.zeros(HT, dtype=np.uint64)
    hstamp = np.zeros(HT, dtype=np.int32)

    for t in range(maxT):
        if firstD >= 0 and t + 1 > firstD + slack:
            break
        hist[:] = 0
        T = 0
        cntAbove = 0
        nc = 0
        rshift = 0
        for i in range(nb):
            Li = bL[i]
            Ai = bA[i]
            nH = axis_options(Li, Ai, 0, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
            nD = axis_options(Li, Ai, 1, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
            nE = axis_options(Li, Ai, 2, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
            for kH in range(nH):
                for kD in range(nD):
                    for kE in range(nE):
                        ok = True
                        full = True
                        cnt = 0
                        pot = 0
                        for w in range(NW):
                            one0 = optOne[0, kH, w]; two0 = optTwo[0, kH, w]
                            one1 = optOne[1, kD, w]; two1 = optTwo[1, kD, w]
                            one2 = optOne[2, kE, w]; two2 = optTwo[2, kE, w]
                            a0 = one0 | two0; a1 = one1 | two1; a2 = one2 | two2
                            ee = ((two0 & ~a1 & ~a2) | (two1 & ~a0 & ~a2) | (two2 & ~a0 & ~a1) |
                                  (one0 & one1 & ~a2) | (one0 & one2 & ~a1) | (one1 & one2 & ~a0))
                            if ee & Sm[w]:
                                ok = False
                                break
                            gg = (two0 & (a1 | a2)) | (two1 & (a0 | a2)) | (two2 & (a0 | a1)) | (one0 & one1 & one2)
                            x = ((bB[i, w] & ~gg) | ee) & Pm[w]
                            if x != Pm[w]:
                                full = False
                            cnt += popcnt(x)
                            pot += popcnt((a0 | a1 | a2) & Pm[w] & ~x)
                        if not ok:
                            continue
                        act = optAct[0, kH] | optAct[1, kD] | optAct[2, kE]
                        if full:
                            if nEnd >= MAXE:
                                continue
                            apply_action(Li, Ai, act, nL, nA)
                            h = config_hash(nL, nA)
                            dup = False
                            for q in range(nEnd):
                                if eHash[q] == h:
                                    dup = True
                                    break
                            if dup:
                                continue
                            eHash[nEnd] = h
                            eDepth[nEnd] = t + 1
                            for e in range(6):
                                eL[nEnd, e] = nL[e]
                                eA[nEnd, e] = nA[e]
                            ePath[nEnd, t] = act
                            j = i
                            for tt in range(t - 1, -1, -1):
                                ePath[nEnd, tt] = actArr[tt, j]
                                j = parArr[tt, j]
                            nEnd += 1
                            if firstD < 0:
                                firstD = t + 1
                            continue
                        key = iB * cnt + iP * pot
                        if key < T:
                            continue
                        hist[key] += 1
                        cntAbove += 1
                        while cntAbove - hist[T] >= CAP:
                            cntAbove -= hist[T]
                            T += 1
                            rshift = 0
                        if key == T and rshift > 0:
                            if (np.random.randint(0, 1 << 30) & ((1 << rshift) - 1)) != 0:
                                continue
                        if nc >= CAPS:
                            # compact: drop keys < T; halve the threshold bucket (uniform subsample)
                            rshift += 1
                            m = 0
                            for q in range(nc):
                                kq = cKey[q]
                                if kq > T or (kq == T and np.random.randint(0, 2) == 0):
                                    cPar[m] = cPar[q]; cAct[m] = cAct[q]; cKey[m] = kq
                                    m += 1
                            nc = m
                            if nc >= CAPS:
                                continue
                        cPar[nc] = i
                        cAct[nc] = act
                        cKey[nc] = key
                        nc += 1
        if firstD >= 0 and t + 1 >= firstD + slack:
            break
        m = 0
        for q in range(nc):
            if cKey[q] >= T:
                cPar[m] = cPar[q]; cAct[m] = cAct[q]; cKey[m] = cKey[q]
                m += 1
        nc = m
        if nc == 0:
            break
        if jit > 0:
            kf = np.empty(nc, dtype=np.float64)
            for q in range(nc):
                kf[q] = -(cKey[q] + np.random.random() * 0.999)
            order = np.argsort(kf)
        else:
            order = np.argsort(-cKey[:nc], kind='mergesort')
        ns = 0
        for qi in range(min(nc, MAXC)):
            q = order[qi]
            i = cPar[q]
            apply_action(bL[i], bA[i], cAct[q], nL, nA)
            stamp(nL, nA, lineBits, NW, e2, g3)
            for w in range(NW):
                nbw[w] = ((bB[i, w] & ~g3[w]) | e2[w]) & Pm[w]
            h = canon_hash(nL, nA, nbw, NW, 0)
            slot = np.int64(h & np.uint64(HT - 1))
            dup = False
            while hstamp[slot] == t + 1:
                if htab[slot] == h:
                    dup = True
                    break
                slot = (slot + 1) & (HT - 1)
            if dup:
                continue
            hstamp[slot] = t + 1
            htab[slot] = h
            psum, pmax = penalty(nL, nA, nbw, Pm, NW, nP, cellLine, nl, tau1, tau2)
            for e in range(6):
                sL[ns, e] = nL[e]
                sA[ns, e] = nA[e]
            for w in range(NW):
                sB[ns, w] = nbw[w]
            sPar[ns] = i
            sAct[ns] = cAct[q]
            sScore[ns] = cKey[q] - wS * psum - wM * pmax + np.random.random() * 1e-3
            ns += 1
        order2 = np.argsort(-sScore[:ns])
        nb = min(ns, W)
        for k in range(nb):
            s = order2[k]
            for e in range(6):
                bL[k, e] = sL[s, e]
                bA[k, e] = sA[s, e]
            for w in range(NW):
                bB[k, w] = sB[s, w]
            parArr[t, k] = sPar[s]
            actArr[t, k] = sAct[s]
    return nEnd, eDepth[:nEnd], eL[:nEnd], eA[:nEnd], ePath[:nEnd]
