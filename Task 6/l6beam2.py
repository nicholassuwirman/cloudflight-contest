"""Beam search with a 1-ply lookahead bonus (improved-plan-2 section 5.2.2).

Identical to l6beam.beam_search, except that in stage 2 the best max(LX, W) deduplicated candidates get
wL * iB * (best safe one-tick gain over all of their children) added to their score. Candidates outside that
set are not selected.
"""
import numpy as np
from numba import njit
from l6beam import (ZERO, popcnt, axis_options, apply_action, stamp, canon_hash, penalty)


@njit(cache=True)
def lookahead(L, A, built, lineBits, Pm, Sm, NW, nl, optAct, optOne, optTwo, optCode):
    """Max over safe children of (new P cells - destroyed P cells); nP+1 if a child completes P."""
    nH = axis_options(L, A, 0, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
    nD = axis_options(L, A, 1, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
    nE = axis_options(L, A, 2, nl, lineBits, NW, optAct, optOne, optTwo, optCode)
    best = 0
    for kH in range(nH):
        for kD in range(nD):
            for kE in range(nE):
                ok = True
                g = 0
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
                    g += popcnt(ee & Pm[w] & ~built[w]) - popcnt(gg & built[w] & ~ee)
                if ok and g > best:
                    best = g
    return best


@njit(cache=True)
def axis_options_lock(L, A, ax, nl, lineBits, NW, optAct, optOne, optTwo, optCode, same):
    """axis_options, restricted to u1 == u2 when same (the pair moves/toggles together)."""
    e1 = ax
    e2 = ax + 3
    n = 0
    for u1 in range(4):
        l1 = L[e1] + (1 if u1 == 1 else 0) - (1 if u1 == 2 else 0)
        a1 = A[e1] ^ (1 if u1 == 3 else 0)
        if l1 < 0 or l1 >= nl:
            continue
        for u2 in range(4):
            if same and u2 != u1:
                continue
            l2 = L[e2] + (1 if u2 == 1 else 0) - (1 if u2 == 2 else 0)
            a2 = A[e2] ^ (1 if u2 == 3 else 0)
            if l2 < 0 or l2 >= nl:
                continue
            c1 = l1 * 2 + a1
            c2 = l2 * 2 + a2
            code = c1 * 64 + c2 if c1 <= c2 else c2 * 64 + c1
            dup = False
            for k in range(n):
                if optCode[ax, k] == code:
                    dup = True
                    break
            if dup:
                continue
            optCode[ax, n] = code
            optAct[ax, n] = (u1 << (2 * e1)) | (u2 << (2 * e2))
            for w in range(NW):
                b1 = lineBits[ax, l1, w] if a1 else ZERO
                b2 = lineBits[ax, l2, w] if a2 else ZERO
                optOne[ax, n, w] = b1 ^ b2
                optTwo[ax, n, w] = b1 & b2
            n += 1
    return n


@njit(cache=True)
def chord_count(built, Pm, lineBits, NW, nl):
    """min over axes of the number of distinct lines that hold a missing P cell."""
    best = 1 << 30
    for ax in range(3):
        n = 0
        for l in range(nl):
            for w in range(NW):
                if lineBits[ax, l, w] & Pm[w] & ~built[w]:
                    n += 1
                    break
        if n < best:
            best = n
    return best


@njit(cache=True)
def beam_search2(lineBits, Pm, Sm, NW, nP, cellLine, nl, K,
                 L0, A0, built0, acc0,
                 W, CAP, maxT, iB, iP, wS, wM, seed, jit, wL, LX, lockAx, lockT, wC):
    """Returns (depth, actions[depth]) of the first found path reaching K acceptances, or (-1, empty)."""
    np.random.seed(seed)
    # beam storage
    bL = np.zeros((W, 6), dtype=np.int64)
    bA = np.zeros((W, 6), dtype=np.int64)
    bB = np.zeros((W, NW), dtype=np.uint64)
    bAcc = np.zeros(W, dtype=np.int64)
    nb = 1
    for e in range(6):
        bL[0, e] = L0[e]
        bA[0, e] = A0[e]
    for w in range(NW):
        bB[0, w] = built0[w]
    bAcc[0] = acc0
    parArr = np.zeros((maxT, W), dtype=np.int32)
    actArr = np.zeros((maxT, W), dtype=np.int32)

    KB = iB * (nP + 1) + iP * (nP + 1) + 1
    maxKey = (K + 1) * KB + 1
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
    sAcc = np.zeros(MAXC, dtype=np.int64)
    sPar = np.zeros(MAXC, dtype=np.int32)
    sAct = np.zeros(MAXC, dtype=np.int32)
    sScore = np.zeros(MAXC, dtype=np.float64)
    HT = 1
    while HT < 4 * MAXC:
        HT *= 2
    htab = np.zeros(HT, dtype=np.uint64)
    hstamp = np.zeros(HT, dtype=np.int32)
    lOpt = np.zeros((3, 16), dtype=np.int64)
    lOne = np.zeros((3, 16, NW), dtype=np.uint64)
    lTwo = np.zeros((3, 16, NW), dtype=np.uint64)
    lCode = np.zeros((3, 16), dtype=np.int64)

    for t in range(maxT):
        hist[:] = 0
        T = 0
        cntAbove = 0
        nc = 0
        rshift = 0
        for i in range(nb):
            Li = bL[i]
            Ai = bA[i]
            lk = t < lockT
            nH = axis_options_lock(Li, Ai, 0, nl, lineBits, NW, optAct, optOne, optTwo, optCode, lk and lockAx == 0)
            nD = axis_options_lock(Li, Ai, 1, nl, lineBits, NW, optAct, optOne, optTwo, optCode, lk and lockAx == 1)
            nE = axis_options_lock(Li, Ai, 2, nl, lineBits, NW, optAct, optOne, optTwo, optCode, lk and lockAx == 2)
            accI = bAcc[i]
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
                            nbw[w] = x
                            if x != Pm[w]:
                                full = False
                            cnt += popcnt(x)
                            pot += popcnt((a0 | a1 | a2) & Pm[w] & ~x)
                        if not ok:
                            continue
                        if full:
                            acc = accI + 1
                            cnt = 0
                            pot = 0
                            for w in range(NW):
                                pass
                            key = acc * KB
                        else:
                            acc = accI
                            key = acc * KB + iB * cnt + iP * pot
                        if key < T:
                            continue
                        act = optAct[0, kH] | optAct[1, kD] | optAct[2, kE]
                        if acc >= K:
                            # found: reconstruct
                            depth = t + 1
                            path = np.zeros(depth, dtype=np.int32)
                            path[t] = act
                            j = i
                            for tt in range(t - 1, -1, -1):
                                path[tt] = actArr[tt, j]
                                j = parArr[tt, j]
                            return depth, path
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
        # stage 2: candidates with key >= T, sorted desc
        m = 0
        for q in range(nc):
            if cKey[q] >= T:
                cPar[m] = cPar[q]; cAct[m] = cAct[q]; cKey[m] = cKey[q]
                m += 1
        nc = m
        if nc == 0:
            return -1, np.zeros(0, dtype=np.int32)
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
            full = True
            cnt = 0
            for w in range(NW):
                x = ((bB[i, w] & ~g3[w]) | e2[w]) & Pm[w]
                nbw[w] = x
                if x != Pm[w]:
                    full = False
            acc = bAcc[i]
            if full:
                acc += 1
                for w in range(NW):
                    nbw[w] = ZERO
            h = canon_hash(nL, nA, nbw, NW, acc)
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
            for w in range(NW):
                cnt += popcnt(nbw[w])
            psum, pmax = penalty(nL, nA, nbw, Pm, NW, nP, cellLine, nl, tau1, tau2)
            for e in range(6):
                sL[ns, e] = nL[e]
                sA[ns, e] = nA[e]
            for w in range(NW):
                sB[ns, w] = nbw[w]
            sAcc[ns] = acc
            sPar[ns] = i
            sAct[ns] = cAct[q]
            sScore[ns] = cKey[q] - wS * psum - wM * pmax + np.random.random() * 1e-3
            if wC > 0.0:
                sScore[ns] -= wC * chord_count(nbw, Pm, lineBits, NW, nl)
            ns += 1
        if wL > 0.0:
            order2 = np.argsort(-sScore[:ns])
            nx = min(ns, max(LX, W))
            for k in range(nx):
                s = order2[k]
                g = lookahead(sL[s], sA[s], sB[s], lineBits, Pm, Sm, NW, nl, lOpt, lOne, lTwo, lCode)
                sScore[s] += wL * iB * g
            for k in range(nx, ns):
                sScore[order2[k]] -= 1e9
        order2 = np.argsort(-sScore[:ns])
        nb = min(ns, W)
        for k in range(nb):
            s = order2[k]
            for e in range(6):
                bL[k, e] = sL[s, e]
                bA[k, e] = sA[s, e]
            for w in range(NW):
                bB[k, w] = sB[s, w]
            bAcc[k] = sAcc[s]
            parArr[t, k] = sPar[s]
            actArr[t, k] = sAct[s]
    return -1, np.zeros(0, dtype=np.int32)
