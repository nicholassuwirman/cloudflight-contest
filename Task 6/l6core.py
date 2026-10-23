"""Level 6 core: parsing, geometry, reference simulator (faithful port of sandbox JS), output I/O."""
import os

DIR_TO = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]  # (dr, dc) E SE SW W NW NE
FIRE = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]    # emitter t fires FIRE[t]
TOKENS = ['-', 'F', 'B', 'T']


def parse_input(path):
    with open(path) as f:
        lines = f.read().split('\n')
    n = int(lines[0].strip())
    idx = 1
    cases = []
    for _ in range(n):
        while not lines[idx].strip():
            idx += 1
        R, rows, cols, K = map(int, lines[idx].split())
        grid = [lines[idx + 1 + i].rstrip('\r') for i in range(rows)]
        idx += 1 + rows
        cases.append({'R': R, 'rows': rows, 'cols': cols, 'K': K, 'grid': grid})
    return cases


class Geometry:
    """Board geometry for board radius R (sandbox radius = R+1)."""

    def __init__(self, R):
        self.R = R
        rad = R + 1
        self.rows = 2 * rad - 1
        self.cols = 4 * rad - 3
        # perimeter (sandbox rebuildPerimeter): start (c=0, r=rad-1), dirs NE,E,SE,SW,W,NW
        per = []
        r, c = rad - 1, 0
        for (dr, dc) in [(-1, 1), (0, 2), (1, 1), (1, -1), (0, -2), (-1, -1)]:
            for _ in range(rad - 1):
                per.append((r, c))
                r, c = r + dr, c + dc
        self.perimeter = per
        self.P = len(per)  # 6R
        cols = self.cols

        def in_grid(c, r):
            if r < 0 or r >= 2 * rad - 1:
                return False
            if (r + c + rad + 1) % 2 != 0:
                return False
            if r < rad:
                return c + r >= rad - 1 and c - r <= cols - rad + 1
            return c + (2 * rad - 1 - r) >= rad - 1 and c + r - 2 * rad + 1 < cols - rad + 1
        self.in_grid = in_grid
        base = []
        for r in range(self.rows):
            row = []
            rrail = r == 0 or r == self.rows - 1
            for c in range(cols):
                if (r + c + rad + 1) % 2 != 0:
                    row.append('-')
                    continue
                if r < rad:
                    crail = c == rad - 1 - r or c == cols - 1 - rad + 1 + r
                    if c + r >= rad - 1 and c - r <= cols - rad + 1:
                        row.append('#' if (rrail or crail) else 'O')
                    else:
                        row.append(':')
                else:
                    crail = c == rad - 1 + r - 2 * rad + 2 or c == cols - 1 + rad - 1 - r
                    if c + 2 * rad - 1 - r >= rad - 1 and c + r - 2 * rad + 1 < cols - rad + 1:
                        row.append('#' if (rrail or crail) else 'O')
                    else:
                        row.append(':')
            base.append(row)
        self.base = base
        # ray cells per (perimeter idx, track)
        self.ray = []
        for (pr, pc) in per:
            lst = []
            for (dr, dc) in FIRE:
                cells = []
                r2, c2 = pr + dr, pc + dc
                while in_grid(c2, r2):
                    cells.append((r2, c2))
                    r2, c2 = r2 + dr, c2 + dc
                lst.append(cells)
            self.ray.append(lst)
        # interior cells
        self.cells = [(r, c) for r in range(self.rows) for c in range(cols) if base[r][c] == 'O']
        self.cell_index = {rc: i for i, rc in enumerate(self.cells)}


_GEOM = {}


def geom(R):
    if R not in _GEOM:
        _GEOM[R] = Geometry(R)
    return _GEOM[R]


def simulate_ref(case, program, stop_at=None):
    """Faithful port of the sandbox. Returns list of 1-based ticks on which a pattern was accepted."""
    g = geom(case['R'])
    pg = case['grid']
    rows, cols = g.rows, g.cols
    assert rows == case['rows'] and cols == case['cols']
    grid = [row[:] for row in g.base]
    idx = [i * g.R for i in range(6)]
    act = [False] * 6
    n = g.P
    # pattern cells and skirt
    pm, skirt = [], []
    for r in range(rows):
        for c in range(cols):
            ch = pg[r][c]
            if ch == 'M':
                pm.append((r, c))
            elif ch == 'O':
                for dr, dc in DIR_TO:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and pg[rr][cc] == 'M':
                        skirt.append((r, c))
                        break
    accepted = []
    for t, line in enumerate(program):
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v == 'x' or v == 'd':
                    grid[r][c] = 'O'
                elif v == 'm':
                    grid[r][c] = 'M'
        for e in range(6):
            ins = line[e]
            if ins == 'F':
                idx[e] = (idx[e] + 1) % n
            elif ins == 'B':
                idx[e] = (idx[e] + n - 1) % n
            elif ins == 'T':
                act[e] = not act[e]
        hit = {}
        for e in range(6):
            if not act[e]:
                continue
            for rc in g.ray[idx[e]][e]:
                hit[rc] = hit.get(rc, 0) + 1
        for (r, c), k in hit.items():
            v = grid[r][c]
            if v != 'O' and v != 'M':
                continue
            if k == 2:
                grid[r][c] = 'm' if v == 'O' else 'M'
            elif k >= 3:
                grid[r][c] = 'd'
        ok = all(grid[r][c] in 'Mm' for r, c in pm) and not any(grid[r][c] in 'Mm' for r, c in skirt)
        if ok:
            for r, c in pm:
                grid[r][c] = 'x'
            accepted.append(t + 1)
            if stop_at is not None and len(accepted) >= stop_at:
                break
    return accepted


def read_output(path):
    with open(path) as f:
        txt = f.read()
    progs, cur = [], []
    for ln in txt.split('\n'):
        if ln.strip():
            cur.append(ln.split())
        elif cur:
            progs.append(cur)
            cur = []
    if cur:
        progs.append(cur)
    return progs


def write_output(path, programs):
    out = []
    for p in programs:
        out.extend(' '.join(l) for l in p)
        out.append('')
    with open(path, 'w', newline='\n') as f:
        f.write('\n'.join(out) + '\n')


def validate_file(in_path, out_path, verbose=True):
    cases = parse_input(in_path)
    progs = read_output(out_path)
    assert len(progs) == len(cases), (len(progs), len(cases))
    total = 0
    bad = 0
    for i, (c, p) in enumerate(zip(cases, progs)):
        for l in p:
            assert len(l) == 6 and all(x in TOKENS for x in l), l
        acc = simulate_ref(c, p)
        if len(acc) != c['K'] or acc[-1] != len(p):
            bad += 1
            if verbose:
                print(f'  case {i}: K={c["K"]} accepted={acc} lines={len(p)}')
        total += len(p)
    return total, bad
