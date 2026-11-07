"""
Dynamic-gesture engine: trajectory template matching.

Static poses answer "what shape is the hand"; this module answers "what
path did it draw". A fingertip trail (positions in hand-size units) is
resampled to a fixed point count, normalized for position and scale, and
compared point-by-point against letter/gesture path templates - the classic
$1-recognizer approach, kept direction-aware (a Z drawn backwards is not a
Z) but chirality-tolerant (the camera is not mirrored, so users draw
letters that arrive horizontally flipped; both hands must work).

Used for the two ASL letters that are traced in the air rather than held
(J and Z), and for oscillation gestures (Wave) via the reversal counter
that also powers HESAP LUTFEN.
"""

import numpy as np

N_POINTS = 32

# Path templates in unit-square coordinates (x right, y DOWN, screen-style).
LETTER_PATHS = {
    # Z: across the top, diagonal to bottom-left, across the bottom
    "Z": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    # J: straight down, then hook through the bottom toward the side
    "J": [(1.0, 0.0), (1.0, 0.55), (0.95, 0.8), (0.8, 0.95), (0.55, 1.0),
          (0.3, 0.95), (0.15, 0.8)],
}

MATCH_THRESHOLD = 0.22   # mean point distance in normalized units


def _resample(path, n=N_POINTS):
    """Path resampled to n points equally spaced along its arc length."""
    # copy=True: the walk below mutates pts in place, and asarray would
    # alias (and corrupt) a caller-owned float32 array
    pts = np.array(path, dtype=np.float32, copy=True)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-6:
        return None
    step = total / (n - 1)
    out = [pts[0]]
    acc = 0.0
    i = 0
    while len(out) < n and i < len(seg):
        if acc + seg[i] >= step - 1e-9:
            t = (step - acc) / max(seg[i], 1e-9)
            new = pts[i] + t * (pts[i + 1] - pts[i])
            out.append(new)
            pts[i] = new
            seg[i] = float(np.linalg.norm(pts[i + 1] - new))
            acc = 0.0
        else:
            acc += seg[i]
            i += 1
    while len(out) < n:
        out.append(pts[-1])
    return np.array(out[:n], dtype=np.float32)


def _normalize(pts):
    """Center on the centroid, scale the longer side to 1."""
    pts = pts - pts.mean(axis=0)
    scale = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-6)
    return pts / scale


def path_distance(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def match(trail, templates=None, threshold=MATCH_THRESHOLD):
    """Best (name, distance) for a drawn trail; ("", inf) when nothing is
    close enough. Tries each template and its horizontal mirror."""
    templates = templates if templates is not None else LETTER_PATHS
    pts = _resample(list(trail))
    if pts is None:
        return "", float("inf")
    pts = _normalize(pts)
    best, best_d = "", float("inf")
    for name, tpl in templates.items():
        t = _normalize(_resample(tpl))
        mirrored = t * np.array([-1.0, 1.0], dtype=np.float32)
        d = min(path_distance(pts, t),
                path_distance(pts, _normalize(mirrored)))
        if d < best_d:
            best, best_d = name, d
    if best_d > threshold:
        return "", best_d
    return best, best_d


def travel(trail):
    """Total arc length of a trail (same units as the trail)."""
    pts = np.asarray(list(trail), dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def oscillation(values, min_step=0.02):
    """(reversals, amplitude) of a 1-D signal - the shared primitive behind
    Wave and the HESAP LUTFEN money-rub."""
    arr = np.asarray(list(values), dtype=np.float32)
    if len(arr) < 3:
        return 0, 0.0
    steps = np.diff(arr)
    steps = steps[np.abs(steps) > min_step]
    if len(steps) < 2:
        return 0, float(np.ptp(arr))
    reversals = int(np.sum(np.sign(steps[1:]) != np.sign(steps[:-1])))
    return reversals, float(np.ptp(arr))
