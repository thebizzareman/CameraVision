"""
Word-sign engine: temporal sequence recognition for whole signs.

Fingerspelling (signs.py) spells words letter by letter. Real sign
languages are mostly *word* signs: a handshape made at a body location,
often with a movement, sometimes two hands. This module recognizes those.

A word sign is a detector of kind "sequence" in gesture_db.json:

  { "type": "word_sign", "language": "ASL",
    "detector": { "kind": "sequence", "hands": 1, "window": 45,
      "phases": [
        {"fingers": {...}, "region": "forehead"},
        {"fingers": {...}, "move": "out"}
      ] } }

Each phase asserts a handshape (finger states, matched by gesture_db), an
optional body location (forehead / face / chin / chest), and an optional
movement relative to the previous phase (out / in / up / down / left /
right, measured against the face). A per-hand state machine advances
through the phases; completing the last one within `window` frames emits
the sign. Locations and movements are face-relative, so signs work at any
distance and position in frame. Purely data-driven - new signs are JSON,
no code (the research pipeline can grow them like everything else).
"""

from collections import deque

import numpy as np

import gesture_db
import motion

MOVE_MIN = 0.35   # displacement (face-widths) that counts as a movement
TRAIL_LEN = 24    # hand-center history for motion-pattern phases


def _classify_motion(trail):
    """Name the pattern a hand-center trail traces: 'circle' (both axes
    oscillate around a loop), 'nod' (repeated vertical), 'shake' (repeated
    horizontal), or None. Trail points are in hand-size units, so it is
    scale-invariant.

    Deliberately strict: a real sign motion covers real distance and
    reverses direction several times. A held pose (even one drifting with
    natural tremor or tracking jitter) has a short path and few reversals,
    so it must classify as None - otherwise a static thumbs-up at the chest
    would read as SORRY, which in ASL requires an actual circling motion."""
    if len(trail) < 14:
        return None
    arr = np.asarray(trail, dtype=np.float32)
    # total path length: deliberate motion covers ground, jitter does not
    path = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
    if path < 1.6:                       # hand-size units
        return None
    xr, yr = float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1]))
    revx, _ = motion.oscillation(arr[:, 0], min_step=0.12)
    revy, _ = motion.oscillation(arr[:, 1], min_step=0.12)
    # circle: both axes swing widely and reverse, with comparable extents
    if (revx >= 2 and revy >= 2 and xr > 0.5 and yr > 0.5
            and 0.4 < (xr / max(yr, 1e-6)) < 2.5):
        return "circle"
    if revy >= 3 and yr > 0.6 and yr > 1.6 * xr:
        return "nod"
    if revx >= 3 and xr > 0.6 and xr > 1.6 * yr:
        return "shake"
    return None


def _region_point(face, region):
    if face is None:
        return None
    if region == "forehead":
        return face.forehead
    if region == "chin":
        return face.chin
    if region == "chest":
        return face.chin + np.array([0.0, face.height * 1.2])
    return face.center   # "face"


class WordSignMatcher:
    """Advances per-hand state machines over sequence detectors and emits
    completed word-sign names."""

    def __init__(self, entries=None):
        self.language = None   # None = all languages; else e.g. "ASL", "TID"
        self.load(entries or [])
        # handedness -> {"sign_id": ..., "phase": int, "anchor": pt,
        #                "age": frames}
        self._state = {}
        # handedness -> deque of recent hand centers (hand-size units), for
        # motion-pattern phases (circle / nod / shake)
        self._trails = {}

    def load(self, entries):
        self._all = [e for e in entries
                     if e.get("status", "active") == "active"
                     and e.get("detector", {}).get("kind") == "sequence"]
        self._apply_language()
        return len(self.signs)

    def set_language(self, language):
        """Restrict recognition to one sign language (plus universal signs).
        None recognizes every language."""
        self.language = language
        self._apply_language()
        self._state = {}
        return len(self.signs)

    def _apply_language(self):
        if self.language is None:
            self.signs = list(self._all)
        else:
            self.signs = [e for e in self._all
                          if e.get("language", "universal")
                          in (self.language, "universal")]

    def _phase_ok(self, phase, hand, face, anchor, trail=None):
        # handshape
        if not gesture_db._match_fingers(phase.get("fingers", {}), hand):
            return False
        # location
        region = phase.get("region")
        if region:
            pt = _region_point(face, region)
            if pt is None:
                return False
            if (float(np.linalg.norm(hand.center - pt))
                    > float(phase.get("max_dist", 1.4)) * _face_w(face)):
                return False
        # motion pattern (circle / nod / shake) traced by the hand center
        want_motion = phase.get("motion")
        if want_motion:
            if trail is None or _classify_motion(trail) != want_motion:
                return False
        # movement since the anchor (previous satisfied phase)
        move = phase.get("move")
        if move and anchor is not None and face is not None:
            d = (hand.center - anchor) / _face_w(face)
            fc = (hand.center - face.center) / _face_w(face)
            anchor_fc = (anchor - face.center) / _face_w(face)
            if move == "down" and d[1] < MOVE_MIN:
                return False
            if move == "up" and -d[1] < MOVE_MIN:
                return False
            if move == "right" and d[0] < MOVE_MIN:
                return False
            if move == "left" and -d[0] < MOVE_MIN:
                return False
            if move == "out" and (np.linalg.norm(fc)
                                  - np.linalg.norm(anchor_fc)) < MOVE_MIN:
                return False
            if move == "in" and (np.linalg.norm(anchor_fc)
                                 - np.linalg.norm(fc)) < MOVE_MIN:
                return False
        return True

    def update(self, hands, face=None):
        """Feed one frame; return a list of (sign_name, handedness) tuples
        for signs completed this frame (handedness attributes each sign to
        the hand that made it, so the recognizer can arbitrate it against
        that hand's other observations)."""
        completed = []
        present = set()
        for hand in hands:
            key = hand.handedness
            present.add(key)
            trail = self._trails.setdefault(key, deque(maxlen=TRAIL_LEN))
            trail.append(tuple(hand.center / max(hand.size, 1.0)))
            st = self._state.get(key)

            # try to advance an in-progress sequence
            if st is not None:
                st["age"] += 1
                sign = _by_id(self.signs, st["sign_id"])
                if sign is None or st["age"] > _window(sign):
                    self._state.pop(key, None)
                    st = None
                else:
                    phases = sign["detector"]["phases"]
                    nxt = phases[st["phase"]]
                    if self._phase_ok(nxt, hand, face, st["anchor"], trail):
                        st["phase"] += 1
                        st["anchor"] = hand.center.copy()
                        st["age"] = 0
                        if st["phase"] >= len(phases):
                            completed.append((sign["name"], key))
                            self._state.pop(key, None)
                            continue

            # not mid-sequence: see if this hand starts any sign's phase 0
            if key not in self._state:
                for sign in self.signs:
                    if sign["detector"].get("hands", 1) != 1:
                        continue
                    phases = sign["detector"]["phases"]
                    if self._phase_ok(phases[0], hand, face, None, trail):
                        if len(phases) == 1:
                            completed.append((sign["name"], key))
                        else:
                            self._state[key] = {
                                "sign_id": sign["id"], "phase": 1,
                                "anchor": hand.center.copy(), "age": 0}
                        break

        for gone in set(self._state) - present:
            self._state.pop(gone, None)
        for gone in set(self._trails) - present:
            self._trails.pop(gone, None)
        return completed


def _face_w(face):
    return max(getattr(face, "width", 1.0), 1.0) if face is not None else 1.0


def _window(sign):
    return int(sign["detector"].get("window", 45))


def _by_id(signs, sign_id):
    for s in signs:
        if s["id"] == sign_id:
            return s
    return None
