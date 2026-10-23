"""Per-pattern compact model: slider line space, bitsets over relevant cells (pattern + skirt)."""
import numpy as np
from l6core import geom, DIR_TO


class LineModel:
    """Per-radius: for emitter t, offset o in [-R, R] -> normalized line index l in [0, 2R]."""

    def __init__(self, R):
        g = geom(R)
        self.R = R
        self.g = g
        nl = 2 * R + 1
        self.nl = nl
        self.sign = [0] * 6
        coord_of = [lambda r, c: r, lambda r, c: c - r, lambda r, c: c + r]
        self.coord_vals = []
        for a in range(3):
            vals = set()
            for t in (a, a + 3):
                for o in range(-R, R + 1):
                    pr, pc = g.perimeter[(t * R + o) % g.P]
                    vals.add(coord_of[a](pr, pc))
            vals = sorted(vals)
            assert len(vals) == nl, (a, vals)
            self.coord_vals.append(vals)
        # chord cells per axis/line
        self.chord = [[[] for _ in range(nl)] for _ in range(3)]
        for (r, c) in g.cells:
            for a in range(3):
                v = coord_of[a](r, c)
                self.chord[a][self.coord_vals[a].index(v)].append((r, c))
        self.line_of = {}
        for t in range(6):
            a = t % 3
            ls = []
            for o in range(-R, R + 1):
                p = (t * R + o) % g.P
                pr, pc = g.perimeter[p]
                l = self.coord_vals[a].index(coord_of[a](pr, pc))
                ls.append(l)
                cells = sorted(rc for rc in g.ray[p][t] if g.base[rc[0]][rc[1]] == 'O')
                if abs(o) < R:
                    assert cells == sorted(self.chord[a][l]), (t, o)
                else:
                    assert not cells
            d = ls[1] - ls[0]
            assert all(ls[i + 1] - ls[i] == d for i in range(len(ls) - 1)) and abs(d) == 1
            assert ls[R] == R
            self.sign[t] = d
        self.cell_lines = {}
        for (r, c) in g.cells:
            self.cell_lines[(r, c)] = tuple(self.coord_vals[a].index(coord_of[a](r, c)) for a in range(3))


_LM = {}


def line_model(R):
    if R not in _LM:
        _LM[R] = LineModel(R)
    return _LM[R]


class PatternModel:
    def __init__(self, case):
        R = case['R']
        lm = line_model(R)
        self.R, self.lm, self.K = R, lm, case['K']
        pg = case['grid']
        rows, cols = len(pg), len(pg[0])
        P, S = [], []
        for r in range(rows):
            for c in range(cols):
                ch = pg[r][c]
                if ch == 'M':
                    P.append((r, c))
                elif ch == 'O':
                    if any(0 <= r + dr < rows and 0 <= c + dc < cols and pg[r + dr][c + dc] == 'M'
                           for dr, dc in DIR_TO):
                        S.append((r, c))
        for rc in P + S:
            assert lm.g.base[rc[0]][rc[1]] == 'O', rc
        self.P, self.S = P, S
        rel = P + S
        self.nP, self.nS = len(P), len(S)
        nrel = len(rel)
        NW = (nrel + 63) // 64
        self.NW = NW
        ridx = {rc: i for i, rc in enumerate(rel)}

        def bits(cells):
            w = np.zeros(NW, dtype=np.uint64)
            for rc in cells:
                if rc in ridx:
                    i = ridx[rc]
                    w[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
            return w
        nl = lm.nl
        self.lineBits = np.zeros((3, nl, NW), dtype=np.uint64)
        for a in range(3):
            for l in range(nl):
                self.lineBits[a, l] = bits(lm.chord[a][l])
        self.Pm = bits(P)
        self.Sm = bits(S)
        self.cellLine = np.array([lm.cell_lines[rc] for rc in P], dtype=np.int32)
        self.sign = np.array(lm.sign, dtype=np.int32)
