"""
Facial expression analysis built on MediaPipe Face Landmarker.

The model returns 478 face landmarks plus 52 "blendshapes" - calibrated
0..1 activation scores for facial muscles (browInnerUp, jawOpen, tongueOut,
mouthSmileLeft, eyeLookOutLeft, ...). Expressions are read from blendshapes,
gaze direction from the eye-look blendshapes, and the landmark geometry is
exposed for face+hand composite gestures (Eyvallah, Yemezler).
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

import gesture_db

FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "face_landmarker.task")

# Declarative expressions from gesture_db.json (detector kind "face_spec") -
# blendshape thresholds as data, merged by the research pipeline and
# hot-reloadable like spec hand gestures.
SPEC_EXPRESSIONS = []


def reload_spec_expressions():
    """Re-read data-driven expressions from the database. Returns count."""
    global SPEC_EXPRESSIONS
    SPEC_EXPRESSIONS = gesture_db.active_face_specs(gesture_db.load())
    return len(SPEC_EXPRESSIONS)


reload_spec_expressions()

# FaceMesh landmark indices
FOREHEAD, CHIN = 10, 152
CHEEK_R, CHEEK_L = 234, 454
# subject's right eye / left eye
EYE_R_OUT, EYE_R_IN, EYE_R_LOW = 33, 133, 145
EYE_L_OUT, EYE_L_IN, EYE_L_LOW = 263, 362, 374
IRIS_R, IRIS_L = 468, 473

RIGHT_EYE_RING = [33, 160, 158, 133, 153, 144]
LEFT_EYE_RING = [263, 387, 385, 362, 380, 373]
RIGHT_BROW = [70, 63, 105, 66, 107]
LEFT_BROW = [336, 296, 334, 293, 300]
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Expression name -> (test over blendshape dict)
EXPRESSION_RULES = {
    "Eyebrows Raised": lambda b: max(
        b.get("browInnerUp", 0),
        (b.get("browOuterUpLeft", 0) + b.get("browOuterUpRight", 0)) / 2) > 0.5,
    # brow-down co-activates when squinting/smiling, so a frown also
    # requires the smile muscles to be quiet
    "Frown": lambda b: (
        (b.get("browDownLeft", 0) + b.get("browDownRight", 0)) / 2 > 0.5
        and (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2 < 0.3
    ),
    "Smile": lambda b: (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2 > 0.4,
    "Mouth Open": lambda b: b.get("jawOpen", 0) > 0.4,
    "Tongue Out": lambda b: b.get("tongueOut", 0) > 0.4,
    "Kiss": lambda b: b.get("mouthPucker", 0) > 0.75,
    "Wink": lambda b: (
        (b.get("eyeBlinkLeft", 0) > 0.55) != (b.get("eyeBlinkRight", 0) > 0.55)
        and min(b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0)) < 0.25
    ),
}


# ---------------------------------------------------------------------------
# Emotion layer (facial_gestures_analysis.md, FACS-based).
# Blendshapes serve as Action Unit proxies: AU1=browInnerUp, AU2=browOuterUp*,
# AU4=browDown*, AU5=eyeWide*, AU6=cheekSquint*, AU7=eyeSquint*,
# AU12=mouthSmile*, AU14=mouthDimple*, AU15=mouthFrown*, AU20=mouthStretch*,
# AU24=mouthPress*, AU26=jawOpen. MediaPipe computes blendshapes on the
# pose-normalized mesh, which satisfies the head-pose requirement (doc 4.3).
# Each rule returns a 0..1 score; only the single best-scoring emotion is
# reported (R12: emotions are mutually exclusive at the frame level).
# ---------------------------------------------------------------------------

def _avg(b, *names):
    return sum(b.get(n, 0.0) for n in names) / len(names)


def _sym(b, left, right):  # symmetric activation (both sides)
    return min(b.get(left, 0.0), b.get(right, 0.0))


def _asym(b, left, right):  # unilateral activation delta
    return abs(b.get(left, 0.0) - b.get(right, 0.0))


def _score_happy(b):
    """Duchenne smile: AU12 + AU6 (doc 2.1). Cheek raiser must be active."""
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    cheek = _avg(b, "cheekSquintLeft", "cheekSquintRight")
    if smile > 0.4 and cheek > 0.25 and _asym(b, "mouthSmileLeft", "mouthSmileRight") < 0.25:
        return (smile + cheek) / 2
    return 0.0


def _score_fake_smile(b):
    """AU12 without AU6 -> social/fake smile (doc 2.1 detection rule)."""
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    cheek = _avg(b, "cheekSquintLeft", "cheekSquintRight")
    if smile > 0.4 and cheek < 0.12 and _asym(b, "mouthSmileLeft", "mouthSmileRight") < 0.25:
        return smile * 0.9
    return 0.0


def _score_surprised(b):
    """AU1+AU2+AU5+AU26, without lip stretching (doc 3.1 positive branch)."""
    brow = min(b.get("browInnerUp", 0.0),
               _avg(b, "browOuterUpLeft", "browOuterUpRight") + 0.15)
    wide = _avg(b, "eyeWideLeft", "eyeWideRight")
    jaw = b.get("jawOpen", 0.0)
    stretch = _avg(b, "mouthStretchLeft", "mouthStretchRight")
    if brow > 0.3 and wide > 0.22 and jaw > 0.35 and stretch < 0.3:
        return (brow + wide + jaw) / 3
    return 0.0


def _score_scared(b):
    """AU1+AU2+AU5+AU20/AU15 - the horror branch of doc 3.1."""
    brow = b.get("browInnerUp", 0.0)
    wide = _avg(b, "eyeWideLeft", "eyeWideRight")
    negative = max(_avg(b, "mouthStretchLeft", "mouthStretchRight"),
                   _avg(b, "mouthFrownLeft", "mouthFrownRight"))
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    if brow > 0.3 and wide > 0.22 and negative > 0.3 and smile < 0.2:
        return (brow + wide + negative) / 3
    return 0.0


def _score_angry(b):
    """AU4 + AU7 + pressed/depressed mouth (doc 1.1 + 3.3 contrast)."""
    down = _avg(b, "browDownLeft", "browDownRight")
    tight = _avg(b, "eyeSquintLeft", "eyeSquintRight")
    mouth = max(_avg(b, "mouthPressLeft", "mouthPressRight"),
                _avg(b, "mouthFrownLeft", "mouthFrownRight"))
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    if down > 0.5 and tight > 0.25 and mouth > 0.3 and smile < 0.2:
        return (down + tight + mouth) / 3
    return 0.0


def _score_concentration(b):
    """AU4 + AU7 with a NEUTRAL mouth - what separates it from anger (3.3)."""
    down = _avg(b, "browDownLeft", "browDownRight")
    tight = _avg(b, "eyeSquintLeft", "eyeSquintRight")
    press = _avg(b, "mouthPressLeft", "mouthPressRight")
    frown = _avg(b, "mouthFrownLeft", "mouthFrownRight")
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    if down > 0.4 and tight > 0.25 and press < 0.25 and frown < 0.25 and smile < 0.2:
        return (down + tight) / 2 * 0.9
    return 0.0


def _score_suspicious(b):
    """Unilateral AU2 + AU7 (doc 2.2): asymmetry is the key signal."""
    brow_delta = _asym(b, "browOuterUpLeft", "browOuterUpRight")
    raised = max(b.get("browOuterUpLeft", 0.0), b.get("browOuterUpRight", 0.0))
    tight = _avg(b, "eyeSquintLeft", "eyeSquintRight")
    if brow_delta > 0.25 and raised > 0.3 and tight > 0.15:
        return (brow_delta + raised + tight) / 3
    return 0.0


def _score_contempt(b):
    """Unilateral AU12/AU14 (doc 2.3): the only strictly one-sided emotion."""
    smile_delta = max(_asym(b, "mouthSmileLeft", "mouthSmileRight"),
                      _asym(b, "mouthDimpleLeft", "mouthDimpleRight"))
    peak = max(b.get("mouthSmileLeft", 0.0), b.get("mouthSmileRight", 0.0),
               b.get("mouthDimpleLeft", 0.0), b.get("mouthDimpleRight", 0.0))
    if smile_delta > 0.25 and peak > 0.35:
        return (smile_delta + peak) / 2
    return 0.0


def _score_suppressed_laugh(b):
    """AU12 fighting AU24 (doc 3.2): pressed lips against smile muscles."""
    press = _avg(b, "mouthPressLeft", "mouthPressRight")
    smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
    if press > 0.4 and smile > 0.25:
        return (press + smile) / 2
    return 0.0


EMOTION_RULES = {
    "Happy": _score_happy,
    "Fake Smile": _score_fake_smile,
    "Surprised": _score_surprised,
    "Scared": _score_scared,
    "Angry": _score_angry,
    "Concentration": _score_concentration,
    "Suspicious": _score_suspicious,
    "Contempt": _score_contempt,
    "Suppressed Laugh": _score_suppressed_laugh,
}
EMOTION_NAMES = set(EMOTION_RULES)
EMOTION_MIN_SCORE = 0.3


def pick_emotion(blendshapes):
    """Top-1 emotion (R12: frame-level exclusivity). Returns (name, score)
    or (None, 0.0) when nothing clears the minimum score."""
    best, best_score = None, EMOTION_MIN_SCORE
    for name, scorer in EMOTION_RULES.items():
        s = scorer(blendshapes)
        if s > best_score:
            best, best_score = name, s
    return best, (best_score if best else 0.0)


class FaceState:
    """One detected face: pixel landmarks, blendshapes, derived features."""

    def __init__(self, landmarks, blendshapes, frame_shape):
        h, w = frame_shape[:2]
        self.points = np.array([(lm.x * w, lm.y * h) for lm in landmarks],
                               dtype=np.float32)
        self.blendshapes = blendshapes  # name -> score
        self.chin = self.points[CHIN]
        self.forehead = self.points[FOREHEAD]
        self.center = (self.chin + self.forehead) / 2
        self.width = float(np.linalg.norm(self.points[CHEEK_R] - self.points[CHEEK_L]))
        self.height = float(np.linalg.norm(self.forehead - self.chin))
        # (under-eye point, eye width) per eye, for the Yemezler gesture
        self.eyes = [
            (self.points[EYE_R_LOW],
             float(np.linalg.norm(self.points[EYE_R_OUT] - self.points[EYE_R_IN]))),
            (self.points[EYE_L_LOW],
             float(np.linalg.norm(self.points[EYE_L_OUT] - self.points[EYE_L_IN]))),
        ]
        self.expressions = {name for name, rule in EXPRESSION_RULES.items()
                            if rule(blendshapes)}
        self.expressions |= {e["name"] for e in SPEC_EXPRESSIONS
                             if gesture_db.match_face_spec(e, blendshapes)}
        self.emotion, self.emotion_score = pick_emotion(blendshapes)
        self.gaze = self._gaze()

    def _gaze(self):
        b = self.blendshapes
        left = (b.get("eyeLookOutLeft", 0) + b.get("eyeLookInRight", 0)) / 2
        right = (b.get("eyeLookOutRight", 0) + b.get("eyeLookInLeft", 0)) / 2
        up = (b.get("eyeLookUpLeft", 0) + b.get("eyeLookUpRight", 0)) / 2
        down = (b.get("eyeLookDownLeft", 0) + b.get("eyeLookDownRight", 0)) / 2
        parts = []
        if up > 0.45 and up > down:
            parts.append("up")
        elif down > 0.55 and down > up:
            parts.append("down")
        if left > 0.4 and left > right:
            parts.append("left")
        elif right > 0.4 and right > left:
            parts.append("right")
        return "-".join(parts) if parts else "center"


class FaceAnalyzer:
    def __init__(self, model_path=FACE_MODEL_PATH):
        if not os.path.isfile(model_path):
            raise ImportError(f"face landmark model missing: {model_path}")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr):
        """Returns a FaceState for the most prominent face, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        blend = {}
        if result.face_blendshapes:
            blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return FaceState(result.face_landmarks[0], blend, frame_bgr.shape)

    def close(self):
        self._landmarker.close()


def draw_face(frame, face):
    pts = face.points.astype(int)
    for ring, color in ((RIGHT_EYE_RING, (0, 255, 180)),
                        (LEFT_EYE_RING, (0, 255, 180)),
                        (RIGHT_BROW, (255, 200, 0)),
                        (LEFT_BROW, (255, 200, 0)),
                        (OUTER_LIPS, (120, 120, 255))):
        cv2.polylines(frame, [pts[ring]], True, color, 1, cv2.LINE_AA)
    for iris in (IRIS_R, IRIS_L):
        cv2.circle(frame, tuple(pts[iris]), 2, (255, 255, 255), -1)

    label = " | ".join(sorted(face.expressions)) if face.expressions else "neutral"
    if face.emotion:
        label = f"{face.emotion.upper()} ({face.emotion_score:.0%})  " + label
    label += f"  [gaze: {face.gaze}]"
    x = int(face.center[0] - face.width * 0.7)
    y = int(pts[:, 1].min() - 12)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
