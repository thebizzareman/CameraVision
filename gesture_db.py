"""
Gesture knowledge database: loader + declarative detection engine.

gesture_db.json holds every gesture the program knows. Entries come in two
detector kinds:
  - "code": implemented in gestures.py / faces.py; the DB entry is the
    catalog record (metadata, display config, finger signature for duplicate
    detection).
  - "spec": detected directly from data by match_spec() below - no code
    change needed to add one. This is what research.py produces.

Spec detector fields:
  fingers:       {"thumb"|"index"|"middle"|"ring"|"pinky":
                  "ext"|"curl"|"half"|"any"} - "half" means partly hooked
                  (continuous curl between 0.25 and 0.8)
  orientation:   optional "tips_up" - fingertips must be above the wrist
  palm:          optional "toward"|"away" - palm orientation vs the camera
  spread:        optional {"pair": [fA, fB], "min_ratio": 1.6} - the two
                 tips must be separated by min_ratio x their MCP gap
                 (Vulcan salute: middle/ring split)
  crossed:       optional [fA, fB] - the two fingers cross (lateral tip
                 order flips relative to the knuckle order)
  thumb_between: optional [fA, fB] - thumb tip pokes through that knuckle
                 pair (generalized fig sign)
  tip_contact:   optional {"fingers": [fA, fB], "max_dist": 0.35} - the two
                 tips touch (within max_dist hand-sizes)
  face_relation: optional {"region": "forehead"|"face"|"chest",
                           "max_dist": <face-widths>} - hand center must be
                  within max_dist face-widths of the region point
  overrides:     builtin gesture names to suppress on the same hand when this
                 spec matches (e.g. Finger Gun overrides Pointing, because
                 the builtin ignores the thumb)

Two-hand spec entries (type "two_hand", kind "spec"):
  hand_a/hand_b: finger-state dicts as above, tried in both hand orders
  relation:      "touching" (centers within max_dist hand-sizes) or
                 "stacked" (hand_a above hand_b, laterally aligned)
"""

import json
import os

import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "gesture_db.json")

FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")


def load(path=DB_PATH):
    """Returns the list of entries; [] on missing/corrupt file so the app
    always starts."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("entries", [])
    except (OSError, ValueError) as exc:
        print(f"gesture_db: could not load {path} ({exc}); using builtins only")
        return []


def save(entries, path=DB_PATH):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc["entries"] = entries
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def active_specs(entries, gesture_type=None):
    out = [e for e in entries
           if e.get("status") == "active"
           and e.get("detector", {}).get("kind") == "spec"]
    if gesture_type:
        out = [e for e in out if e.get("type") == gesture_type]
    return out


def active_face_specs(entries):
    """Declarative facial-expression entries (detector kind 'face_spec')."""
    return [e for e in entries
            if e.get("status") == "active"
            and e.get("detector", {}).get("kind") == "face_spec"]


def blend_value(blendshapes, key):
    """Blendshape score with pair shorthand: 'mouthSmile' averages
    mouthSmileLeft/mouthSmileRight when the bare key doesn't exist."""
    if key in blendshapes:
        return float(blendshapes[key])
    left = blendshapes.get(key + "Left")
    right = blendshapes.get(key + "Right")
    if left is None and right is None:
        return 0.0
    return (float(left or 0.0) + float(right or 0.0)) / 2.0


def match_face_spec(entry, blendshapes):
    """True when the face's blendshapes satisfy the entry's thresholds:
    every 'blends' key at or above its minimum, every 'quiet' key below
    its maximum."""
    det = entry.get("detector", {})
    blends = det.get("blends")
    if not blends:
        return False
    for key, min_v in blends.items():
        if blend_value(blendshapes, key) < float(min_v):
            return False
    for key, max_v in det.get("quiet", {}).items():
        if blend_value(blendshapes, key) > float(max_v):
            return False
    return True


def signature_of(entry):
    """Normalized finger signature tuple for duplicate detection, or None."""
    det = entry.get("detector", {})
    sig = det.get("signature") or det.get("fingers")
    if not sig:
        return None
    return tuple(sig.get(k, "any") for k in FINGER_KEYS)


def signatures_collide(a, b):
    """Two signatures collide when every finger state is compatible
    ('any' matches everything)."""
    if a is None or b is None:
        return False
    return all(x == y or "any" in (x, y) for x, y in zip(a, b))


def _finger_states(hand):
    idx, mid, ring, pinky = hand.fingers_extended
    return {"thumb": hand.thumb_extended, "index": idx, "middle": mid,
            "ring": ring, "pinky": pinky}


def _finger_curl(hand, key):
    """Continuous curl 0..1 for one finger, from Hand analytics."""
    if key == "thumb":
        return float(getattr(hand, "thumb_curl", 1.0))
    curls = getattr(hand, "finger_curls", None)
    if curls is None:
        return 1.0
    return float(curls[FINGER_KEYS.index(key) - 1])


# landmark indices per finger (match gestures.py)
_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
_PIP = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def _match_fingers(fingers, hand):
    states = _finger_states(hand)
    for key, want in fingers.items():
        if want == "any":
            continue
        if want == "ext" and not states.get(key, False):
            return False
        if want == "curl" and states.get(key, False):
            return False
        if want == "half" and not 0.25 <= _finger_curl(hand, key) <= 0.8:
            return False
    return True


def _d(hand, a, b):
    return float(np.linalg.norm(hand.points[a] - hand.points[b]))


def match_spec(entry, hand, face=None):
    """True when `hand` (gestures.Hand) satisfies the entry's spec detector.
    face is a faces.FaceState, required only for face_relation specs."""
    det = entry.get("detector", {})
    fingers = det.get("fingers")
    if not fingers:
        return False
    if not _match_fingers(fingers, hand):
        return False

    spread = det.get("spread")
    if spread:
        fa, fb = spread["pair"]
        tip_gap = _d(hand, _TIP[fa], _TIP[fb])
        mcp_gap = max(_d(hand, _MCP[fa], _MCP[fb]), 1e-3)
        if tip_gap / mcp_gap < float(spread.get("min_ratio", 1.6)):
            return False

    crossed = det.get("crossed")
    if crossed:
        fa, fb = crossed
        lat = hand.points[_MCP["index"]] - hand.points[_MCP["pinky"]]
        n = float(np.linalg.norm(lat)) or 1.0
        lat = lat / n
        mcp_d = float(np.dot(hand.points[_MCP[fb]] - hand.points[_MCP[fa]], lat))
        tip_d = float(np.dot(hand.points[_TIP[fb]] - hand.points[_TIP[fa]], lat))
        if mcp_d * tip_d >= 0:
            return False

    between = det.get("thumb_between")
    if between:
        fa, fb = between
        mid = (hand.points[_PIP[fa]] + hand.points[_PIP[fb]]) / 2.0
        near = float(np.linalg.norm(hand.points[_TIP["thumb"]] - mid))
        if near > 0.45 * max(hand.size, 1.0):
            return False
        wrist = hand.points[0]
        knuckle = (hand.points[_PIP["index"]] + hand.points[_PIP["pinky"]]) / 2.0
        beyond = (float(np.linalg.norm(hand.points[_TIP["thumb"]] - wrist))
                  > float(np.linalg.norm(knuckle - wrist)) * 0.95)
        if not beyond:
            return False

    contact = det.get("tip_contact")
    if contact:
        fa, fb = contact["fingers"]
        max_d = float(contact.get("max_dist", 0.35)) * max(hand.size, 1.0)
        if _d(hand, _TIP[fa], _TIP[fb]) > max_d:
            return False

    if det.get("orientation") == "tips_up":
        tips = hand.points[[8, 12, 16, 20]]  # index..pinky fingertips
        if float(np.mean(tips[:, 1])) >= float(hand.points[0][1]):
            return False  # fingertips not above the wrist

    palm = det.get("palm")   # "toward" | "away" (camera-relative)
    if palm in ("toward", "away"):
        facing = getattr(hand, "palm_facing", None)
        if facing is None or facing != (palm == "toward"):
            return False

    rel = det.get("face_relation")
    if rel:
        if face is None:
            return False
        region = rel.get("region", "face")
        if region == "forehead":
            point = face.forehead
        elif region == "chest":
            point = face.chin + np.array([0.0, face.height * 1.2])
        else:
            point = face.center
        max_dist = float(rel.get("max_dist", 1.0)) * face.width
        if float(np.linalg.norm(hand.center - point)) > max_dist:
            return False

    return True


def match_two_hand(entry, hand_a, hand_b):
    """True when the two hands satisfy a two-hand spec entry, tried in
    both orderings."""
    det = entry.get("detector", {})
    if det.get("kind") != "spec":
        return False
    spec_a, spec_b = det.get("hand_a"), det.get("hand_b")
    if not spec_a or not spec_b:
        return False
    relation = det.get("relation", "touching")
    max_dist = float(det.get("max_dist", 1.0))

    def fits(ha, hb):
        if not (_match_fingers(spec_a, ha) and _match_fingers(spec_b, hb)):
            return False
        ref = max(ha.size, hb.size, 1.0)
        dx = abs(float(ha.center[0] - hb.center[0]))
        dy = float(hb.center[1] - ha.center[1])   # >0: hand_a is higher
        if relation == "stacked":
            return 0.2 * ref < dy < 3.0 * ref and dx < 1.6 * ref
        dist = float(np.linalg.norm(ha.center - hb.center))
        return dist < max_dist * ref

    return fits(hand_a, hand_b) or fits(hand_b, hand_a)
