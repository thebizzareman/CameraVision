"""
ASL fingerspelling recognition (sign language layer).

Classifies the 24 static letters of the American manual alphabet from the
continuous hand analytics computed by gestures.Hand (per-joint curl values,
finger directions, tip geometry). J and Z are traced in the air (dynamic)
and are not detected.

Architecture: soft template scoring, not hard rules. Every letter has a
score function built from tolerant 0..1 components (how extended is the
index, how close is the thumb to the middle PIP, ...) combined with a
geometric mean - so an imperfectly-held letter degrades gracefully to a
lower score instead of falling through to "undetected", and lookalike
letters are separated by whichever template fits best. All positional
tests run in hand-axis coordinates (along/across the palm), so letters
work at any hand rotation, not just bolt upright.

  classify(hand)        -> best letter or "" (below acceptance threshold)
  classify_scored(hand) -> (letter, score 0..1)
  Speller               -> commits stable letters into a spelled word

Sign mode is toggled with the L key in the app. While it is on, the hands
belong to the speller: RULEBOOK.md R14 suppresses all hand-gesture
observations so letter shapes (V/Peace, Y/Shaka, T/Nah, F/OK...) are not
double-reported. Facial expressions stay live.
"""

import cv2
import numpy as np

# Landmark indices (match gestures.py)
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

SUPPORTED = "ABCDEFGHIKLMNOPQRSTUVWXY"  # J and Z are dynamic

ACCEPT = 0.5    # minimum score to commit a letter (below this, shown only
                # as a dim live guess so the user still sees feedback)
GUESS_MIN = 0.4  # show a dim best-guess chip down to this score


# ---------------------------------------------------------------------------
# Score components: every helper maps a measurement to 0..1
# ---------------------------------------------------------------------------

def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _ext(c):
    """Finger reads straight (curl near 0)."""
    return _clip01((0.55 - c) / 0.4)


def _fold(c):
    """Finger reads folded (curl near 1)."""
    return _clip01((c - 0.4) / 0.35)


def _half(c):
    """Finger reads half-hooked."""
    return _clip01(1.0 - abs(c - 0.5) / 0.32)


def _near(x, target, tol):
    return _clip01(1.0 - abs(x - target) / tol)


def _low(x, hi):
    """1 when x is 0, fading to 0 at hi."""
    return _clip01(1.0 - x / hi)


def _gm(*xs):
    """Geometric mean of clipped components: one dead component kills the
    letter, several weak ones drag it down smoothly."""
    p = 1.0
    for x in xs:
        p *= _clip01(x)
    return p ** (1.0 / len(xs))


# ---------------------------------------------------------------------------
# Feature extraction (rotation-invariant hand-axis frame)
# ---------------------------------------------------------------------------

def _features(hand):
    pts = hand.points
    palm_w = float(np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP]))
    scale = max(hand.size, palm_w, 1.0)

    axis = pts[MIDDLE_MCP] - pts[WRIST]           # palm "up" direction
    axis = axis / (float(np.linalg.norm(axis)) or 1.0)
    lat = pts[INDEX_MCP] - pts[PINKY_MCP]          # toward the index side
    lat = lat / (float(np.linalg.norm(lat)) or 1.0)

    def h(i):
        """Height along the palm axis, in scale units (wrist=0, knuckles~1)."""
        return float(np.dot(pts[i] - pts[WRIST], axis)) / scale

    def latc(i):
        """Sideways coordinate, positive toward the index edge."""
        return float(np.dot(pts[i] - pts[WRIST], lat)) / scale

    def d(a, b):
        return float(np.linalg.norm(pts[a] - pts[b])) / scale

    f = {
        "scale": scale, "axis": axis, "lat": lat, "h": h, "latc": latc,
        "d": d,
        "ct": float(hand.thumb_curl),
        "ci": hand.finger_curls[0], "cm": hand.finger_curls[1],
        "cr": hand.finger_curls[2], "cp": hand.finger_curls[3],
        "pinch_i": d(THUMB_TIP, INDEX_TIP),
        "pinch_m": d(THUMB_TIP, MIDDLE_TIP),
        "gap_im": d(INDEX_TIP, MIDDLE_TIP),
        "spread_ip": d(INDEX_TIP, PINKY_TIP),
        "t_h": h(THUMB_TIP),
        "t_mpip": d(THUMB_TIP, MIDDLE_PIP),
        # thumb sideways position past the index edge (positive = beside
        # the fist like A/Y, negative = crossing the palm like S/E)
        "t_proj": float(np.dot(pts[THUMB_TIP] - pts[INDEX_MCP], lat)) / scale,
        "tips_h": (h(INDEX_TIP) + h(MIDDLE_TIP) + h(RING_TIP)) / 3.0,
    }
    f["avg_c"] = (f["ci"] + f["cm"] + f["cr"] + f["cp"]) / 4.0

    # 3D thumb depth, oriented so positive always means "tucked behind the
    # fingers" from the signer's perspective regardless of palm direction.
    # 0.0 (synthetic hands / no depth) keeps every depth component neutral.
    depth = float(getattr(hand, "thumb_depth", 0.0))
    if depth and not getattr(hand, "palm_facing", True):
        depth = -depth
    f["t_depth"] = depth

    # pointing direction (screen coords) of the index+middle pair when they
    # are extended, else the palm axis - used for up/horizontal/down letters
    if f["ci"] < 0.55:
        v = hand.finger_dir(0)
        if f["cm"] < 0.55:
            v = v + hand.finger_dir(1)
    else:
        v = axis
    n = float(np.linalg.norm(v)) or 1.0
    f["dir"] = v / n
    f["up"] = _clip01((-f["dir"][1] - 0.25) / 0.5)
    f["down"] = _clip01((f["dir"][1] - 0.25) / 0.5)
    f["horiz"] = _clip01((abs(f["dir"][0]) - abs(f["dir"][1]) + 0.2) / 0.5)

    # index/middle crossing (R): sideways order of the tips flips vs MCPs
    mcp_d = float(np.dot(pts[MIDDLE_MCP] - pts[INDEX_MCP], lat))
    tip_d = float(np.dot(pts[MIDDLE_TIP] - pts[INDEX_TIP], lat))
    f["crossed"] = bool(mcp_d * tip_d < 0)

    # angle between thumb and index pointing directions (L vs G)
    iv, tv = hand.finger_dir(0), hand.thumb_dir
    f["ang_ti"] = float(np.degrees(np.arccos(
        np.clip(float(np.dot(iv, tv)), -1.0, 1.0))))

    # closest folded-knuckle pair to the thumb tip (T/N/M poke test)
    tt = pts[THUMB_TIP]
    pairs = {"T": (INDEX_PIP, MIDDLE_PIP), "N": (MIDDLE_PIP, RING_PIP),
             "M": (RING_PIP, PINKY_PIP)}
    dists = {k: float(np.linalg.norm(tt - (pts[a] + pts[b]) / 2.0)) / scale
             for k, (a, b) in pairs.items()}
    f["poke_letter"] = min(dists, key=dists.get)
    f["poke_dist"] = dists[f["poke_letter"]]
    return f


# ---------------------------------------------------------------------------
# Letter templates
# ---------------------------------------------------------------------------

def _fist(f):
    return _gm(_fold(f["ci"]), _fold(f["cm"]), _fold(f["cr"]), _fold(f["cp"]))


def _thumb_front(f):
    """Thumb crossing in FRONT of the fingers (S/E). Neutral 0.5 w/o depth."""
    return 0.5 if f["t_depth"] == 0.0 else _clip01(0.5 - f["t_depth"] * 1.1)


def _thumb_behind(f):
    """Thumb tucked BEHIND the fingers (T/N/M). Neutral 0.5 w/o depth."""
    return 0.5 if f["t_depth"] == 0.0 else _clip01(0.5 + f["t_depth"] * 1.1)


def _pair(f):
    """Index+middle out, ring+pinky folded."""
    return _gm(_ext(f["ci"]), _ext(f["cm"]), _fold(f["cr"]), _fold(f["cp"]))


def _index_only(f):
    return _gm(_ext(f["ci"]), _fold(f["cm"]), _fold(f["cr"]), _fold(f["cp"]))


def _poke(f, letter):
    """Thumb tip pushed up between folded knuckles (T/N/M family). The
    height gate separates it from S (thumb crossing lower), the pinch
    guard from O (fingertips meeting the thumb, not wrapping over it)."""
    if f["poke_letter"] != letter:
        return 0.0
    return _gm(_fist(f), _clip01((f["t_h"] - 0.85) / 0.2),
               _low(f["poke_dist"], 0.42),
               _clip01((f["pinch_i"] - 0.22) / 0.15),
               _thumb_behind(f))


LETTER_SCORES = {
    # -- fist family ------------------------------------------------------
    "A": lambda f: _gm(_fist(f), _near(f["t_h"], 1.0, 0.75),
                       _clip01((f["t_proj"] + 0.18) / 0.4),
                       _clip01((f["pinch_i"] - 0.18) / 0.2),
                       _clip01((f["avg_c"] - 0.6) / 0.3)),
    "S": lambda f: _gm(_fist(f), _clip01((0.10 - f["t_proj"]) / 0.35),
                       _near(f["t_h"], 0.62, 0.5),
                       _clip01((f["t_h"] - f["tips_h"] + 0.28) / 0.3),
                       _thumb_front(f)),
    "E": lambda f: _gm(_fist(f), _clip01((0.10 - f["t_proj"]) / 0.35),
                       _low(f["pinch_m"], 0.75),
                       _clip01((f["tips_h"] - f["t_h"] + 0.12) / 0.3),
                       _clip01((f["ci"] - 0.55) / 0.25),
                       _thumb_front(f)),
    "T": lambda f: _poke(f, "T"),
    "N": lambda f: _poke(f, "N"),
    "M": lambda f: _poke(f, "M"),

    # -- curved family ----------------------------------------------------
    "C": lambda f: _gm(_near(f["ci"], 0.5, 0.45), _near(f["cm"], 0.5, 0.45),
                       _near(f["cr"], 0.5, 0.45), _near(f["cp"], 0.5, 0.45),
                       _near(f["pinch_i"], 0.75, 0.55),
                       _clip01((f["pinch_i"] - 0.3) / 0.18),
                       _near(f["ct"], 0.35, 0.55)),
    "O": lambda f: _gm(_near(f["avg_c"], 0.72, 0.4),
                       _low(f["pinch_i"], 0.4), _low(f["pinch_m"], 0.8)),
    "X": lambda f: _gm(_near(f["ci"], 0.5, 0.45),
                       _clip01((f["cm"] - 0.6) / 0.3),
                       _clip01((f["cr"] - 0.6) / 0.3),
                       _clip01((f["cp"] - 0.6) / 0.3),
                       _clip01((f["pinch_i"] - 0.3) / 0.25)),

    # -- pinch family -----------------------------------------------------
    "F": lambda f: _gm(_low(f["pinch_i"], 0.5), _ext(f["cm"]),
                       _ext(f["cr"]), _ext(f["cp"])),
    "D": lambda f: _gm(_ext(f["ci"]), _low(f["pinch_m"], 0.55),
                       _fold(f["cr"]), _fold(f["cp"]),
                       _clip01((f["pinch_i"] - 0.25) / 0.25)),

    # -- flat family ------------------------------------------------------
    "B": lambda f: _gm(_ext(f["ci"]), _ext(f["cm"]), _ext(f["cr"]),
                       _ext(f["cp"]), _low(f["spread_ip"], 1.3),
                       _clip01((f["ct"] - 0.2) / 0.3)),
    "W": lambda f: _gm(_ext(f["ci"]), _ext(f["cm"]), _ext(f["cr"]),
                       _fold(f["cp"]), _low(f["spread_ip"], 2.0)),

    # -- index+middle family ----------------------------------------------
    "U": lambda f: _gm(_pair(f), _low(f["gap_im"], 0.45),
                       0.0 if f["crossed"] else 1.0, f["up"]),
    "H": lambda f: _gm(_pair(f), _low(f["gap_im"], 0.5),
                       0.0 if f["crossed"] else 1.0, f["horiz"]),
    "R": lambda f: _gm(_pair(f), _low(f["gap_im"], 0.5),
                       1.0 if f["crossed"] else 0.0),
    "V": lambda f: _gm(_pair(f), _clip01((f["gap_im"] - 0.28) / 0.22),
                       max(_clip01((f["t_mpip"] - 0.35) / 0.3),
                           _clip01((0.55 - f["t_h"]) / 0.3)),
                       f["up"]),
    "K": lambda f: _gm(_pair(f), _clip01((f["gap_im"] - 0.22) / 0.22),
                       _low(f["t_mpip"], 0.5),
                       _clip01((f["t_h"] - 0.55) / 0.3), f["up"]),
    "P": lambda f: _gm(_pair(f), _low(f["t_mpip"], 0.6), f["down"]),

    # -- index-only family -------------------------------------------------
    "L": lambda f: _gm(_index_only(f), f["up"], _near(f["ct"], 0.15, 0.5),
                       _clip01((f["ang_ti"] - 40) / 30)),
    "G": lambda f: _gm(_index_only(f), f["horiz"], _near(f["ct"], 0.2, 0.55),
                       _clip01((65 - f["ang_ti"]) / 35)),
    "Q": lambda f: _gm(_index_only(f), f["down"], _near(f["ct"], 0.3, 0.55)),

    # -- pinky family ------------------------------------------------------
    "I": lambda f: _gm(_fold(f["ci"]), _fold(f["cm"]), _fold(f["cr"]),
                       _ext(f["cp"]), _clip01((f["ct"] - 0.3) / 0.3)),
    "Y": lambda f: _gm(_fold(f["ci"]), _fold(f["cm"]), _fold(f["cr"]),
                       _ext(f["cp"]), _near(f["ct"], 0.15, 0.5),
                       _clip01((f["t_proj"] + 0.05) / 0.35)),
}


# Per-letter score biases learned from the user's K-key corrections
LETTER_BIAS = {}


def set_bias(bias):
    """Install learned per-letter biases (from config.letter_bias)."""
    global LETTER_BIAS
    LETTER_BIAS = dict(bias or {})


def best_guess(hand):
    """(top letter, score) for this pose, ignoring the ACCEPT gate - the
    classifier's closest match, for live on-screen feedback."""
    f = _features(hand)
    best, best_s = "", 0.0
    for letter, fn in LETTER_SCORES.items():
        s = fn(f) + LETTER_BIAS.get(letter, 0.0)
        if s > best_s:
            best, best_s = letter, s
    return best, best_s


def classify_scored(hand):
    """(best letter, score). Score is 0..1; letter is "" below ACCEPT."""
    best, best_s = best_guess(hand)
    if best_s < ACCEPT:
        return "", best_s
    return best, best_s


def classify(hand):
    """Best ASL letter for this hand pose, or "" when nothing fits."""
    return classify_scored(hand)[0]


class Speller:
    """Commits stable letters into a spelled word.

    A letter types once it has been the classified letter for `commit`
    consecutive frames. To type a double letter, relax the hand briefly
    (or hide it) between the two holds. Losing the hand for `gap` frames
    inserts a space (word boundary)."""

    def __init__(self, commit=8, gap=45, max_len=24):
        self._commit = commit
        self._gap = gap
        self._max_len = max_len
        self.word = ""
        self._current = ""
        self._count = 0
        self._typed_current = False
        self._absent = 0

    def update(self, letter):
        """Feed one frame's letter ("" when no hand / no letter)."""
        if not letter:
            self._absent += 1
            if self._absent >= 4:   # short release re-arms the same letter
                self._current, self._count = "", 0
                self._typed_current = False
            if (self._absent == self._gap and self.word
                    and not self.word.endswith(" ")):
                self.word += " "
            return
        self._absent = 0
        if letter == self._current:
            self._count += 1
        else:
            self._current, self._count = letter, 1
            self._typed_current = False
        if self._count >= self._commit and not self._typed_current:
            self.word = (self.word + letter)[-self._max_len:]
            self._typed_current = True

    def commit_now(self, letter):
        """Type a letter immediately (motion letters J/Z: the trace IS the
        hold, no extra dwell time needed)."""
        self.word = (self.word + letter)[-self._max_len:]
        self._current, self._count = letter, self._commit
        self._typed_current = True
        self._absent = 0

    @property
    def progress(self):
        """0..1 hold progress of the letter currently being typed."""
        if not self._current or self._typed_current:
            return 0.0
        return min(1.0, self._count / self._commit)

    def clear(self):
        self.word = ""
        self._current, self._count = "", 0
        self._typed_current = False
        self._absent = 0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_letters(frame, hands, current="", progress=0.0):
    """Big letter chip beside each signing hand, with a confidence readout
    and - on the letter being typed - a hold-progress ring that fills as
    the dwell time completes (so you can SEE when it will commit)."""
    for hand in hands:
        # committed/stable letter (bright), else the live best guess (dim) so
        # a letter is always visible while signing, not only once confident
        letter = getattr(hand, "sign_letter", "")
        confident = bool(letter)
        if letter:
            score = float(getattr(hand, "sign_score", 0.0))
        else:
            letter = getattr(hand, "sign_guess", "")
            score = float(getattr(hand, "sign_guess_score", 0.0))
        if not letter:
            continue
        border = (255, 210, 60) if confident else (150, 150, 150)
        text_col = (255, 255, 255) if confident else (170, 190, 210)
        pts = hand.points.astype(int)
        x = int(pts[:, 0].max()) + 12
        y = int(pts[:, 1].min())
        x2, y2 = x + 64, y + 72
        if x2 > frame.shape[1] - 2:
            x, x2 = int(pts[:, 0].min()) - 76, int(pts[:, 0].min()) - 12
        roi_y2, roi_x2 = min(y2, frame.shape[0]), min(x2, frame.shape[1])
        if y < 0 or x < 0 or roi_y2 <= y or roi_x2 <= x:
            continue
        roi = frame[y:roi_y2, x:roi_x2]
        cv2.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, roi)
        cv2.rectangle(frame, (x, y), (x2, y2), border, 2)
        cv2.putText(frame, letter, (x + 16, y + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, text_col, 2)
        if score > 0:
            cv2.putText(frame, f"{score:.0%}", (x + 12, y + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1)
        if confident and letter == current and progress > 0:
            center = (x + 32, y + 32)
            cv2.ellipse(frame, center, (34, 34), -90, 0,
                        int(360 * min(progress, 1.0)), (60, 255, 120), 3)


def draw_spelling(frame, word):
    """Spelled-word banner along the bottom edge."""
    text = "ASL: " + (word if word else "-")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    x, y = 12, frame.shape[0] - 18
    roi = frame[y - th - 12:y + 10, x - 6:x + tw + 14]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (60, 210, 255), 2)
