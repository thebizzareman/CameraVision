"""
Hand gesture recognition built on MediaPipe Hands (21 landmarks per hand).

Two layers:
  1. Per-hand gestures  - classified from finger extension states
     (Open Palm, Fist, Thumbs Up, Peace, Pointing, OK, Rock)
  2. Composite gestures - spatial relations between two hands, registered in
     COMPOSITE_GESTURES. "Kapak" = open palm resting on top of a fist.

Results are temporally smoothed so gestures must hold for a few frames
before they fire, which kills flicker and false positives.
"""

import os
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

import config as user_config
from rules import ConflictFilter
from faces import EXPRESSION_RULES, EMOTION_NAMES
import gesture_db
import motion
import signs
import signword

# Per-user calibration (C key writes this; config.json persists it)
USER = user_config.load()
signs.set_bias(USER.get("letter_bias"))

# Facial states are labeled at the face, never as pop-up cards
EXPRESSION_NAMES = set(EXPRESSION_RULES) | EMOTION_NAMES

# Gesture knowledge database (gesture_db.json) - catalog of builtins plus
# declarative 'spec' gestures added by research.py
DB_ENTRIES = gesture_db.load()

# Basic hand poses stay labeled at the hand (like facial expressions)
# instead of raising a pop-up alert card. Driven by the DB's display field.
HAND_LABEL_ONLY = ({e["name"] for e in DB_ENTRIES
                    if e.get("display") == "hand_label"}
                   or {"Open Palm", "Fist", "Rock"})

# Proper on-screen names (Turkish diacritics need the PIL text path)
DISPLAY_NAMES = {e["name"]: e.get("display_name", e["name"])
                 for e in DB_ENTRIES}

HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "hand_landmarker.task")

# MediaPipe hand landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGERS = [  # (mcp, pip, tip) for index..pinky
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


class Hand:
    """One detected hand: pixel-space landmarks plus derived features."""

    def __init__(self, landmarks, handedness, frame_shape):
        h, w = frame_shape[:2]
        self.points = np.array([(lm.x * w, lm.y * h) for lm in landmarks],
                               dtype=np.float32)
        # MediaPipe depth: z per landmark, roughly in normalized-x units,
        # negative = closer to the camera. Zero for synthetic hands.
        self.z = np.array([getattr(lm, "z", 0.0) for lm in landmarks],
                          dtype=np.float32)
        self.has_depth = bool(np.any(np.abs(self.z) > 1e-6))
        self._frame_w = float(w)
        self.handedness = handedness  # "Left" / "Right"
        self.size = float(np.linalg.norm(self.points[MIDDLE_MCP] - self.points[WRIST]))
        self.center = self.points[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP,
                                   PINKY_MCP]].mean(axis=0)
        # Continuous per-finger curl (0.0 straight .. 1.0 fully folded),
        # measured from the bend angle at the PIP joint - rotation invariant
        # and far finer-grained than the old binary extended test.
        self.finger_curls = [self._joint_curl(mcp, pip, tip)
                             for mcp, pip, tip in FINGERS]
        self.thumb_curl = self._thumb_curl()
        self.fingers_extended = self._finger_states()
        self.thumb_extended = self._thumb_state()
        self.gesture = self._classify()

    def _joint_curl(self, a, b, c):
        """Bend at joint b on the a-b-c chain: 0.0 = straight (180 deg),
        1.0 = folded shut (<= 60 deg)."""
        v1 = self.points[a] - self.points[b]
        v2 = self.points[c] - self.points[b]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom < 1e-6:
            return 1.0
        cos = float(np.dot(v1, v2)) / denom
        ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return float(np.clip((180.0 - ang) / 120.0, 0.0, 1.0))

    def _thumb_curl(self):
        """Continuous thumb fold: blends the IP bend angle with how far the
        tip has crossed the palm toward the pinky."""
        angle = self._joint_curl(THUMB_MCP, THUMB_IP, THUMB_TIP)
        anchor = self.points[PINKY_MCP]
        d_tip = float(np.linalg.norm(self.points[THUMB_TIP] - anchor))
        d_mcp = float(np.linalg.norm(self.points[THUMB_MCP] - anchor)) or 1.0
        ratio_curl = float(np.clip((1.35 - d_tip / d_mcp) / 0.5, 0.0, 1.0))
        return 0.5 * angle + 0.5 * ratio_curl

    def _finger_states(self):
        """Extension test per finger: the joint must be nearly straight AND
        the tip farther from the wrist than its PIP joint. The angle gate
        rejects half-hooked fingers; its threshold is per-user calibrated
        (C key) because hands differ."""
        wrist = self.points[WRIST]
        ext_max = float(USER.get("curl_ext_max", 0.5))
        states = []
        for i, (_, pip, tip) in enumerate(FINGERS):
            d_tip = np.linalg.norm(self.points[tip] - wrist)
            d_pip = np.linalg.norm(self.points[pip] - wrist)
            states.append(bool(self.finger_curls[i] < ext_max
                               and d_tip > d_pip * 1.05))
        return states

    def _thumb_state(self):
        """Thumb curls across the palm toward the pinky when folded."""
        anchor = self.points[PINKY_MCP]
        d_tip = np.linalg.norm(self.points[THUMB_TIP] - anchor)
        d_mcp = np.linalg.norm(self.points[THUMB_MCP] - anchor)
        return d_tip > d_mcp * 1.15

    @property
    def palm_facing(self):
        """True when the palm faces the camera. Pure 2D winding test: the
        index->pinky knuckle order flips between palm and back of hand,
        mirrored between left and right hands."""
        i = self.points[INDEX_MCP] - self.points[WRIST]
        p = self.points[PINKY_MCP] - self.points[WRIST]
        cross = float(i[0] * p[1] - i[1] * p[0])
        return (cross > 0) == (self.handedness == "Right")

    @property
    def thumb_depth(self):
        """Thumb tip depth relative to the folded-knuckle plane, in
        hand-size units: positive = tucked behind the fingers (M/N/T),
        negative = crossing in front of them (S/E). 0.0 without depth
        data, which every consumer treats as neutral."""
        if not self.has_depth:
            return 0.0
        pips = float(self.z[[INDEX_PIP, MIDDLE_PIP, RING_PIP,
                             PINKY_PIP]].mean())
        size_norm = max(self.size / self._frame_w, 1e-3)
        return float((self.z[THUMB_TIP] - pips) / size_norm)

    def finger_dir(self, i):
        """Unit pointing direction of finger i (0=index..3=pinky), MCP->tip.
        Screen coordinates: y is negative when pointing up."""
        mcp, _, tip = FINGERS[i]
        v = self.points[tip] - self.points[mcp]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)

    @property
    def thumb_dir(self):
        v = self.points[THUMB_TIP] - self.points[THUMB_MCP]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)

    @property
    def extended_count(self):
        return sum(self.fingers_extended) + (1 if self.thumb_extended else 0)

    def _classify(self):
        idx, mid, ring, pinky = self.fingers_extended
        thumb = self.thumb_extended
        n = sum(self.fingers_extended)

        # OK sign: thumb tip touches index tip, other fingers up
        pinch = (np.linalg.norm(self.points[THUMB_TIP] - self.points[INDEX_TIP])
                 < self.size * 0.35)
        if pinch and mid and ring and pinky:
            return "OK"
        if n == 4:
            return "Open Palm"
        if n == 0:
            # Nah / fig sign: thumb tip pokes out from between the index and
            # middle knuckles - laterally close to them AND beyond them
            knuckle_mid = (self.points[INDEX_PIP] + self.points[MIDDLE_PIP]) / 2
            wrist = self.points[WRIST]
            lateral = np.linalg.norm(self.points[THUMB_TIP] - knuckle_mid)
            beyond = (np.linalg.norm(self.points[THUMB_TIP] - wrist)
                      > np.linalg.norm(knuckle_mid - wrist) * 1.02)
            if lateral < self.size * 0.45 and beyond:
                return "Nah"
            if thumb:
                # Thumbs up: thumb clearly above the fist knuckles
                if self.points[THUMB_TIP][1] < self.points[INDEX_MCP][1] - self.size * 0.3:
                    return "Thumbs Up"
                return "Fist"  # thumb sticking out sideways still reads as fist
            return "Fist"
        if idx and mid and not ring and not pinky:
            return "Peace"
        if idx and not mid and not ring and not pinky:
            return "Pointing"
        if mid and not idx and not ring and not pinky:
            # R4 (RULEBOOK.md): the middle finger must point predominantly
            # upward; a sideways digit is not classified
            v = self.points[MIDDLE_TIP] - self.points[MIDDLE_MCP]
            if v[1] < 0 and abs(v[1]) > abs(v[0]):
                return "Middle Finger"
            return ""
        if idx and pinky and not mid and not ring:
            return "Rock"
        return ""  # no confident single-hand gesture

    @property
    def is_open(self):
        """Open-ish: tolerant of foreshortening when the palm faces down."""
        return sum(self.fingers_extended) >= 3

    @property
    def is_fist(self):
        return sum(self.fingers_extended) == 0


# ---------------------------------------------------------------------------
# Composite (two-hand) gestures. Each entry: name -> predicate(hand_a, hand_b)
# evaluated both ways round, so argument order never matters.
# ---------------------------------------------------------------------------

def _kapak(top, bottom):
    """'Kapak' (Turkish: lid): an open palm resting on top of a fist."""
    if not (top.is_open and bottom.is_fist):
        return False
    ref = max(bottom.size, 1.0)
    dy = bottom.center[1] - top.center[1]          # >0 means top hand is higher
    dx = abs(top.center[0] - bottom.center[0])
    return 0.2 * ref < dy < 3.0 * ref and dx < 1.6 * ref


COMPOSITE_GESTURES = {
    "KAPAK": _kapak,
    # Add more two-hand gestures here, e.g.
    # "TIME OUT": lambda top, bottom: ...
}


# ---------------------------------------------------------------------------
# Face + hand gestures. Each entry: name -> predicate(hand, face) where face
# is a faces.FaceState. Evaluated for every visible hand.
# ---------------------------------------------------------------------------

def _eyvallah(hand, face):
    """'Eyvallah': open palm placed flat on the chest, below the face."""
    if hand.gesture != "Open Palm":
        return False
    dx = abs(hand.center[0] - face.center[0])
    dy = hand.center[1] - face.chin[1]  # positive: below the chin
    return dx < face.width * 1.2 and 0.1 * face.height < dy < 2.4 * face.height


def _yemezler(hand, face):
    """'Yemezler' / goz alti cekme: index fingertip on the skin under an eye."""
    idx, mid, ring, pinky = hand.fingers_extended
    if not idx or mid or ring or pinky:  # index out, other fingers curled
        return False
    tip = hand.points[INDEX_TIP]
    for under_eye, eye_w in face.eyes:
        near = np.linalg.norm(tip - under_eye) < eye_w * 0.9
        below = tip[1] > under_eye[1] - eye_w * 0.3
        if near and below:
            return True
    return False


FACE_HAND_GESTURES = {
    "EYVALLAH": _eyvallah,
    "YEMEZLER": _yemezler,
}


def _money_pose(hand):
    """'Hesap lutfen' static pose: thumb, index and middle tips bunched."""
    t = hand.points[THUMB_TIP]
    i = hand.points[INDEX_TIP]
    m = hand.points[MIDDLE_TIP]
    s = max(hand.size, 1.0)
    _, _, ring, pinky = hand.fingers_extended
    return (np.linalg.norm(t - i) < 0.45 * s
            and np.linalg.norm(t - m) < 0.55 * s
            and np.linalg.norm(i - m) < 0.45 * s
            and not ring and not pinky)


# Special (kilavuzu) gestures get a red alert card; the rest get amber
SPECIAL_GESTURES = (set(COMPOSITE_GESTURES) | set(FACE_HAND_GESTURES)
                    | {"HESAP LUTFEN"})

# Precision arbitration: each hand does ONE thing at a time. Observations are
# tagged with a specificity tier; per hand, only the most specific (lowest
# tier) survives, so a word sign suppresses the generic pose underneath it
# instead of stacking cards. Lower number = more specific = wins.
TIER_WORD = 0        # whole word signs (handshape + location + motion)
TIER_COMPOSITE = 1   # two-hand (KAPAK) and face+hand (EYVALLAH) signs
TIER_DYNAMIC = 2     # motion gestures (HESAP LUTFEN, Wave)
TIER_SPEC = 3        # declarative single-hand handshapes (Shaka, ILY, ...)
TIER_POSE = 4        # builtin static poses (Open Palm, Peace, Fist, ...)
TIER_FACE = 9        # facial expressions/emotions - a separate modality


class GestureRecognizer:
    def __init__(self, max_hands=2, history=7, needed=4,
                 model_path=HAND_MODEL_PATH):
        if not os.path.isfile(model_path):
            raise ImportError(f"hand landmark model missing: {model_path}")
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        # Per-hand thumb-tip motion trail for the money-rub gesture
        self._rub = {}
        # RULEBOOK.md conflict filter - runs on every frame before voting
        self._conflicts = ConflictFilter()
        # Declarative gestures from gesture_db.json (research.py additions)
        self.reload_db()
        # Rolling vote per gesture name -> deque of bools
        self._history = {}
        self._history_len = history
        self._needed = needed
        self._active = set()   # gestures currently held
        self.events = []       # (timestamp_str, gesture_name) rising edges
        self.event_count = 0   # monotonic, survives the 6-entry trim
        # ASL fingerspelling layer (signs.py). While sign_mode is on, the
        # hands belong to the speller and RULEBOOK.md R14 suppresses all
        # hand-gesture observations (face expressions stay live).
        self.sign_mode = False
        self.speller = signs.Speller()
        self._letter_hist = {}   # handedness -> deque of recent raw letters
        self._sign_hand = None   # handedness the speller is following
        # Dynamic layer: fingertip trails for traced letters (J/Z) and
        # hand-center trails for the Wave gesture
        self._sign_trails = {}   # handedness -> deque of tip px positions
        self._wave = {}          # handedness -> deque of center x / size
        # Word-sign layer (signword.py): whole signs like HELLO, THANK YOU.
        # A completed sign is momentary, so it is latched for a few frames
        # to survive the 4-of-7 temporal vote and show its card.
        self._wordsigns = signword.WordSignMatcher(DB_ENTRIES)
        self._wordsign_latch = {}
        # Active sign language for word signs (ASL default). Cycle with the
        # app's language key; TID = Turkish Sign Language.
        self.sign_language = "ASL"
        self._wordsigns.set_language(self.sign_language)

    # sign languages present in the database, plus "All"
    SIGN_LANGUAGES = ("ASL", "TID", None)

    def cycle_language(self):
        """Advance to the next sign language for word signs; returns its
        display label."""
        langs = self.SIGN_LANGUAGES
        i = (langs.index(self.sign_language) + 1) % len(langs)
        return self.set_language(langs[i])

    def set_language(self, language):
        """Set the active sign language for word signs directly (E=ASL,
        T=TID from the app). Returns its display label."""
        self.sign_language = language
        self._wordsigns.set_language(language)
        self._wordsign_latch.clear()
        return language or "All"

    def reload_db(self):
        """Re-read gesture_db.json - lets the app pick up freshly researched
        gestures without a restart. Also reloads data-driven facial
        expressions (face_spec entries) and keeps them labeled at the face
        instead of raising alert cards."""
        global DB_ENTRIES, HAND_LABEL_ONLY, EXPRESSION_NAMES
        DB_ENTRIES = gesture_db.load()
        HAND_LABEL_ONLY = ({e["name"] for e in DB_ENTRIES
                            if e.get("display") == "hand_label"}
                           or {"Open Palm", "Fist", "Rock"})
        import faces
        n_expr = faces.reload_spec_expressions()
        EXPRESSION_NAMES = (set(EXPRESSION_RULES) | EMOTION_NAMES
                            | {e["name"] for e in faces.SPEC_EXPRESSIONS})
        global DISPLAY_NAMES
        DISPLAY_NAMES = {e["name"]: e.get("display_name", e["name"])
                         for e in DB_ENTRIES}
        self._spec_singles = gesture_db.active_specs(DB_ENTRIES, "single_hand")
        self._spec_facehand = gesture_db.active_specs(DB_ENTRIES, "face_hand")
        self._spec_twohand = gesture_db.active_specs(DB_ENTRIES, "two_hand")
        n_words = 0
        if hasattr(self, "_wordsigns"):
            n_words = self._wordsigns.load(DB_ENTRIES)
        return (len(self._spec_singles) + len(self._spec_facehand)
                + len(self._spec_twohand) + n_expr + n_words)

    def process(self, frame_bgr, face=None):
        """Detect hands, classify gestures, update temporal smoothing.
        `face` is an optional faces.FaceState enabling face+hand gestures
        and folding facial expressions into the same voting/event log.
        Returns (hands, active_gestures) with stable gesture names."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode wants strictly increasing timestamps; 33ms ~ 30fps is fine
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands = []
        for lm_list, handed in zip(result.hand_landmarks, result.handedness):
            hands.append(Hand(lm_list, handed[0].category_name,
                              frame_bgr.shape))

        # R14: in sign mode the hands are fingerspelling - classify letters,
        # feed the speller, and raise no hand-gesture observations at all
        # (letter shapes collide with V/Peace, Y/Shaka, T/Nah, F/OK...).
        if self.sign_mode:
            self._process_signs(hands)

        # Raw observations with hand attribution AND a specificity tier, for
        # arbitration. tobs: list of (name, hand_index, tier).
        tobs = []

        def drop(hand_i, name):   # remove a superseded observation
            tobs[:] = [(n, h, t) for (n, h, t) in tobs
                       if not (h == hand_i and n == name)]

        if not self.sign_mode:
            for i, hand in enumerate(hands):
                if hand.gesture:
                    tobs.append((hand.gesture, i, TIER_POSE))
                if self._money_rub(hand):
                    tobs.append(("HESAP LUTFEN", i, TIER_DYNAMIC))
                if self._waving(hand):
                    tobs.append(("Wave", i, TIER_DYNAMIC))
                for entry in self._spec_singles:
                    if gesture_db.match_spec(entry, hand):
                        for ov in entry.get("detector", {}).get("overrides", []):
                            drop(i, ov)
                        tobs.append((entry["name"], i, TIER_SPEC))
            if len(hands) == 2:
                a, b = hands
                for name, check in COMPOSITE_GESTURES.items():
                    if check(a, b) or check(b, a):
                        tobs.append((name, None, TIER_COMPOSITE))
                for entry in self._spec_twohand:
                    if gesture_db.match_two_hand(entry, a, b):
                        tobs.append((entry["name"], None, TIER_COMPOSITE))
            # word-sign sequences - attributed to the hand that made them and
            # latched a few frames to survive the temporal vote
            handed_idx = {h.handedness: i for i, h in enumerate(hands)}
            for name, handed in self._wordsigns.update(hands, face):
                self._wordsign_latch[name] = [12, handed]
            for name in list(self._wordsign_latch):
                self._wordsign_latch[name][0] -= 1
                if self._wordsign_latch[name][0] <= 0:
                    del self._wordsign_latch[name]
                    continue
                idx = handed_idx.get(self._wordsign_latch[name][1])
                tobs.append((name, idx, TIER_WORD))
        if face is not None:
            if not self.sign_mode:
                for i, hand in enumerate(hands):
                    for name, check in FACE_HAND_GESTURES.items():
                        if check(hand, face):
                            tobs.append((name, i, TIER_COMPOSITE))
                    for entry in self._spec_facehand:
                        if gesture_db.match_spec(entry, hand, face):
                            for ov in entry.get("detector", {}).get("overrides", []):
                                drop(i, ov)
                            tobs.append((entry["name"], i, TIER_COMPOSITE))
            for expr in face.expressions:
                tobs.append((expr, None, TIER_FACE))
            if getattr(face, "emotion", None):
                tobs.append((face.emotion, None, TIER_FACE))

        # Precision arbitration: one winner per hand, then the RULEBOOK.md
        # restriction filter, then temporal voting.
        obs = self._arbitrate(tobs)
        seen, notes = self._conflicts.apply(obs, hands, face)
        for note in notes:
            self._log_event("! " + note)

        return hands, self._vote(seen)

    def _arbitrate(self, tobs):
        """Reduce tiered observations to one gesture per hand: the most
        specific wins and suppresses the generic ones underneath it, so
        cards stop stacking. Facial expressions/emotions are a separate
        modality and always pass; a two-hand sign commits both hands and
        suppresses their individual poses. Returns [(name, hand_index)]."""
        face = [(n, h) for (n, h, t) in tobs if t >= TIER_FACE]
        two_hand = [(n, h) for (n, h, t) in tobs
                    if t == TIER_COMPOSITE and h is None]
        if two_hand:
            # both hands belong to the composite; drop every single-hand read
            return face + two_hand
        per_hand = {}
        for (n, h, t) in tobs:
            if t >= TIER_FACE:
                continue
            best = per_hand.get(h)
            if best is None or t < best[1]:
                per_hand[h] = (n, t)
        return face + [(n, h) for h, (n, t) in per_hand.items()]

    def _log_event(self, name):
        import datetime
        self.events.append(
            (datetime.datetime.now().strftime("%H:%M:%S"), name))
        self.events = self.events[-6:]
        self.event_count += 1

    def _process_signs(self, hands):
        """Per-hand letter classification with a short majority vote (kills
        chip flicker on real hands), then feed the speller from the signing
        hand - sticky to the hand it was already following, so a second
        hand entering the frame cannot steal the word mid-spell.

        Motion is resolved BEFORE the static feed: J and Z start from the
        static shapes I (pinky out) and pointing (index out), so a hand that
        is actively tracing a letter must not also commit its static letter
        (else drawing a J first types a spurious I). A still pinky-out hand
        keeps committing I; only once it moves does it become J."""
        # --- motion pass: update trails, detect tracing, commit J/Z --------
        tracing = set()
        committed = {}
        for hand in hands:
            idx, mid, ring, pinky = hand.fingers_extended
            if pinky and not (idx or mid or ring):
                tip, want = hand.points[PINKY_TIP], "J"
            elif idx and not (mid or ring or pinky):
                tip, want = hand.points[INDEX_TIP], "Z"
            else:
                self._sign_trails.pop(hand.handedness, None)
                continue
            trail = self._sign_trails.setdefault(
                hand.handedness, deque(maxlen=40))
            trail.append(tuple(tip / max(hand.size, 1.0)))
            # "moving" (vs a still hold) once the path shows real travel -
            # well below the commit threshold, enough to gate the static I
            if len(trail) >= 8 and motion.travel(trail) > 0.6:
                tracing.add(hand.handedness)
            if len(trail) >= 14 and motion.travel(trail) > 1.6:
                name, _ = motion.match(trail)
                if name == want:
                    self.speller.commit_now(name)
                    committed[hand.handedness] = name
                    trail.clear()
        for gone in set(self._sign_trails) - {h.handedness for h in hands}:
            self._sign_trails.pop(gone, None)

        # --- static pass: score letters, smoothed by a short majority vote -
        scored = {}
        for hand in hands:
            if hand.handedness in committed:
                hand.sign_letter = committed[hand.handedness]
                hand.sign_score = 1.0
                scored[hand.handedness] = ("", 0.0)  # motion owns this hand
                continue
            raw, score = signs.classify_scored(hand)
            hist = self._letter_hist.setdefault(hand.handedness,
                                                deque(maxlen=5))
            hist.append(raw)
            votes = {}
            for l in hist:
                if l:
                    votes[l] = votes.get(l, 0) + 1
            smoothed = max(votes, key=votes.get) if votes else ""
            if smoothed and votes[smoothed] < 3 and len(hist) == hist.maxlen:
                smoothed = ""     # no stable majority yet
            hand.sign_letter = smoothed
            hand.sign_score = score if smoothed == raw else 0.0
            # live best-guess (below the commit threshold) so a letter is
            # always visible as feedback, dim until it is confident enough
            guess, gscore = signs.best_guess(hand)
            hand.sign_guess = guess if gscore >= signs.GUESS_MIN else ""
            hand.sign_guess_score = gscore
            # a hand mid-trace suppresses its static letter feed (I->J, etc.)
            feed = "" if hand.handedness in tracing else smoothed
            scored[hand.handedness] = (feed, score)
        for gone in set(self._letter_hist) - {h.handedness for h in hands}:
            self._letter_hist.pop(gone, None)

        # a hand that just committed a motion letter is done for this frame
        if any(committed):
            return

        # pick the signing hand: keep following the current one while it
        # still shows a letter; otherwise switch to the best-scoring hand
        pick = self._sign_hand
        if not (pick in scored and scored[pick][0]):
            lettered = {k: v for k, v in scored.items() if v[0]}
            pick = (max(lettered, key=lambda k: lettered[k][1])
                    if lettered else None)
        self._sign_hand = pick
        self.speller.update(scored[pick][0] if pick in scored else "")

    def _waving(self, hand):
        """Wave: an open hand rocking side to side. Oscillation of the hand
        center along x - at least 3 direction reversals with real travel."""
        trail = self._wave.setdefault(hand.handedness, deque(maxlen=30))
        if not hand.is_open:
            trail.clear()
            return False
        trail.append(float(hand.center[0]) / max(hand.size, 1.0))
        if len(trail) < 10:
            return False
        reversals, amplitude = motion.oscillation(trail, min_step=0.04)
        return reversals >= 3 and amplitude > 0.7

    def _money_rub(self, hand):
        """'Hesap lutfen': the bunched fingertips pose held while the thumb
        rubs back and forth across the index/middle tips. Tracks the thumb
        tip relative to the hand and looks for oscillation along the
        dominant motion axis."""
        trail = self._rub.setdefault(hand.handedness, deque(maxlen=14))
        if not _money_pose(hand):
            trail.clear()
            return False
        trail.append((hand.points[THUMB_TIP] - hand.center)
                     / max(hand.size, 1.0))
        if len(trail) < 8:
            return False
        arr = np.array(trail)
        centered = arr - arr.mean(axis=0)
        # dominant motion axis (first principal component)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[0]
        if proj.max() - proj.min() < 0.08:  # too little travel: hand is still
            return False
        steps = np.diff(proj)
        steps = steps[np.abs(steps) > 0.012]  # ignore jitter
        if len(steps) < 3:
            return False
        reversals = int(np.sum(np.sign(steps[1:]) != np.sign(steps[:-1])))
        return reversals >= 2

    def _vote(self, seen):
        """A gesture is 'active' once seen in >= needed of the last N frames."""
        all_names = set(self._history) | seen
        active = set()
        for name in all_names:
            hist = self._history.setdefault(
                name, deque(maxlen=self._history_len))
            hist.append(name in seen)
            if sum(hist) >= self._needed:
                active.add(name)

        for name in active - self._active:
            self._log_event(name)
        self._active = active
        return active

    def close(self):
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_hands(frame, hands, active):
    for hand in hands:
        pts = hand.points.astype(int)
        color = (0, 200, 255) if hand.handedness == "Right" else (255, 180, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
        for i, p in enumerate(pts):
            r = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(frame, tuple(p), r, (255, 255, 255), -1)
            cv2.circle(frame, tuple(p), r, color, 1)

        # The hand label names the pose only for hand_label gestures (Open
        # Palm, Fist, Rock); card gestures (Peace, PLEASE, ...) are already
        # named on their card, so the label doesn't duplicate them.
        label = hand.handedness
        letter = getattr(hand, "sign_letter", "")
        if letter:
            label += f": ASL {letter}"
        elif hand.gesture and hand.gesture in HAND_LABEL_ONLY:
            label += f": {hand.gesture}"
        label += f" ({hand.extended_count})"
        x, y = int(hand.center[0] - 60), int(pts[:, 1].min() - 14)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    # facial states stay at the face, basic hand poses at the hand;
    # everything else pops up as an alert card
    draw_alerts(frame, [g for g in active
                        if g not in EXPRESSION_NAMES
                        and g not in HAND_LABEL_ONLY])


def draw_alerts(frame, names):
    """One medium pop-up alert card per active gesture, stacked top-right.
    Kilavuzu gestures get a red card, the rest amber. Names render through
    the Unicode text path so Turkish diacritics display properly."""
    import ui
    y = 66
    card_h = 52
    for name in sorted(names)[:5]:
        disp = DISPLAY_NAMES.get(name, name)
        color = (0, 0, 230) if name in SPECIAL_GESTURES else (0, 165, 255)
        (tw, _), _ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        x2 = frame.shape[1] - 14
        x1 = x2 - (tw + 78)
        y2 = y + card_h
        if y2 > frame.shape[0] - 10 or x1 < 0:
            break
        # darkened translucent card background
        roi = frame[y:y2, x1:x2]
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
        cv2.rectangle(frame, (x1, y), (x2, y2), color, 2)
        # warning-triangle icon with "!"
        cx, cy = x1 + 26, y + card_h // 2
        tri = np.array([[cx, cy - 14], [cx - 15, cy + 12], [cx + 15, cy + 12]])
        cv2.fillPoly(frame, [tri], color)
        cv2.putText(frame, "!", (cx - 5, cy + 9), cv2.FONT_HERSHEY_DUPLEX,
                    0.6, (0, 0, 0), 2)
        ui.text(frame, disp, (x1 + 52, y + 12), size=24)
        y = y2 + 10


def draw_events(frame, events, origin_y):
    y = origin_y
    for ts, name in reversed(events):
        line = f"{ts}  {name}"
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 255, 180), 1)
        y += 24
