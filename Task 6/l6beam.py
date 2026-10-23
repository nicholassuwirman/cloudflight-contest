"""Numba beam search over 6-slider configurations.

Slider state: line index l in [0, 2R] (0 and 2R are dead/silent) + active flag.
Per-slider action u: 0 stay, 1 l+1, 2 l-1, 3 toggle. Action code = sum u_e << (2e).
Emitters e and e+3 share axis e (0=H, 1=D, 2=E).
"""
import numpy as np
from numba import njit

U1 = np.uint64(1)
M1 = np.uint64(0x5555555555555555)
M2 = np.uint64(0x3333333333333333)
M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)
S1 = np.uint64(1)
S2 = np.uint64(2)
S4 = np.uint64(4)
S56 = np.uint64(56)
ZERO = np.uint64(0)


@njit(cache=True, inline='always')
def popcnt(x):
    x = x - ((x >> S1) & M1)
    x = (x & M2) + ((x >> S2) & M2)
    x = (x + (x >> S4)) & M4
    return np.int64((x * H01) >> S56)


@njit(cache=True)
def mix64(h):
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xc4ceb9fe1a85ec53)
    h ^= h >> np.uint64(33)
    return h


@njit(cache=True)
def axis_options(L, A, ax, nl, lineBits, NW, optAct, optOne, optTwo, optCode):
    """Enumerate distinct (canonical) next states of the slider pair of axis ax."""
    e1 = ax
    e2 = ax + 3
    n = 0
    for u1 in range(4):
        l1 = L[e1] + (1 if u1 == 1 else 0) - (1 if u1 == 2 else 0)
        a1 = A[e1] ^ (1 if u1 == 3 else 0)
        if l1 < 0 or l1 >= nl:
            continue
        for u2 in range(4):
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
def apply_action(L, A, act, nL, nA):
    for e in range(6):
        u = (act >> (2 * e)) & 3
        nL[e] = L[e] + (1 if u == 1 else 0) - (1 if u == 2 else 0)
        nA[e] = A[e] ^ (1 if u == 3 else 0)


@njit(cache=True)
def stamp(L, A, lineBits, NW, e2out, g3out):
    for w in range(NW):
        one0 = ZERO; two0 = ZERO; one1 = ZERO; two1 = ZERO; one2 = ZERO; two2 = ZERO
        for ax in range(3):
            b1 = lineBits[ax, L[ax], w] if A[ax] else ZERO
            b2 = lineBits[ax, L[ax + 3], w] if A[ax + 3] else ZERO
            o = b1 ^ b2
            t = b1 & b2
            if ax == 0:
                one0 = o; two0 = t
            elif ax == 1:
                one1 = o; two1 = t
            else:
                one2 = o; two2 = t
        a0 = one0 | two0; a1 = one1 | two1; a2 = one2 | two2
        e2out[w] = ((two0 & ~a1 & ~a2) | (two1 & ~a0 & ~a2) | (two2 & ~a0 & ~a1) |
                    (one0 & one1 & ~a2) | (one0 & one2 & ~a1) | (one1 & one2 & ~a0))
        g3out[w] = (two0 & (a1 | a2)) | (two1 & (a0 | a2)) | (two2 & (a0 | a1)) | (one0 & one1 & one2)


@njit(cache=True)
def canon_hash(L, A, built, NW, acc):
    h = np.uint64(acc) * np.uint64(0x9E3779B97F4A7C15)
    for ax in range(3):
        c1 = L[ax] * 2 + A[ax]
        c2 = L[ax + 3] * 2 + A[ax + 3]
        code = c1 * 64 + c2 if c1 <= c2 else c2 * 64 + c1
        h = mix64(h ^ np.uint64(code + 4096 * ax + 1))
    for w in range(NW):
        h = mix64(h ^ built[w])
    return h


@njit(cache=True)
def penalty(L, A, built, Pm, NW, nP, cellLine, nl, tau1, tau2):
    for ax in range(3):
        for l in range(nl):
            d1 = abs(l - L[ax]) + (1 - A[ax])
            d2 = abs(l - L[ax + 3]) + (1 - A[ax + 3])
            tau1[ax, l] = min(d1, d2)
            tau2[ax, l] = max(d1, d2)
    psum = 0
    pmax = 0
    for p in range(nP):
        if (built[p >> 6] >> np.uint64(p & 63)) & U1:
            continue
        h = cellLine[p, 0]
        d = cellLine[p, 1]
        e = cellLine[p, 2]
        th = tau1[0, h]; td = tau1[1, d]; te = tau1[2, e]
        v = tau2[0, h]
        x = tau2[1, d]
        if x < v:
            v = x
        x = tau2[2, e]
        if x < v:
            v = x
        x = max(th, td)
        if x < v:
            v = x
        x = max(th, te)
        if x < v:
            v = x
        x = max(td, te)
        if x < v:
            v = x
        if v < 1:
            v = 1
        psum += v
        if v > pmax:
            pmax = v
    return psum, pmax


@njit(cache=True)
def beam_search(lineBits, Pm, Sm, NW, nP, cellLine, nl, K,
                L0, A0, built0, acc0,
                W, CAP, maxT, iB, iP, wS, wM, seed, jit):
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

    for t in range(maxT):
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
            bAcc[k] = sAcc[s]
            parArr[t, k] = sPar[s]
            actArr[t, k] = sAct[s]
    return -1, np.zeros(0, dtype=np.int32)
