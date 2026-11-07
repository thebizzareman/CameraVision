"""
Per-user configuration and learned personalization (config.json).

Two learning loops write here:
  - C-key calibration measures YOUR open-palm and fist curl values and
    places the extension threshold between them (hands differ; one-size
    thresholds misread short or long fingers).
  - K-key corrections nudge per-letter score biases whenever you tell the
    speller "that letter was wrong, I meant X" - the classifier adapts to
    your hand instead of you adapting to the classifier.

The file is plain JSON next to the app; delete it to reset all learning.
"""

import json
import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "config.json")

# Everything the program has LEARNED, for carrying between machines: the
# per-user calibration + letter biases (config.json), the correction log,
# and the grown gesture database (builtin + researched entries).
LEARNING_FILES = ["config.json", "corrections.jsonl", "gesture_db.json"]

DEFAULTS = {
    "curl_ext_max": 0.5,   # a finger counts as extended below this curl
    "letter_bias": {},     # ASL letter -> additive score bias (learned)
    "voice": False,        # speak spelled words / fired gestures
}

BIAS_STEP = 0.02
BIAS_LIMIT = 0.10


def load():
    cfg = {}
    try:
        with open(PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass
    out = dict(DEFAULTS)
    out.update(cfg)
    return out


def save(cfg):
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PATH)


def calibrate_threshold(open_mean, fist_mean):
    """Extension threshold from measured open-palm / fist curls: 45% of
    the way from open to fist, clamped to a sane band."""
    if fist_mean <= open_mean + 0.15:
        return DEFAULTS["curl_ext_max"]   # bad capture: keep the default
    t = open_mean + 0.45 * (fist_mean - open_mean)
    return float(min(0.65, max(0.35, t)))


def apply_correction(cfg, shown, intended):
    """Nudge letter biases after a user correction. Returns the bias map."""
    bias = cfg.setdefault("letter_bias", {})
    if intended:
        bias[intended] = min(BIAS_LIMIT,
                             bias.get(intended, 0.0) + BIAS_STEP)
    if shown and shown != intended:
        bias[shown] = max(-BIAS_LIMIT, bias.get(shown, 0.0) - BIAS_STEP)
    return bias


# ---------------------------------------------------------------------------
# Portable learning: carry calibration, corrections and the grown database
# between machines as a single zip.
# ---------------------------------------------------------------------------

def export_learning(zip_path):
    """Bundle everything the program has learned into a zip. Returns the
    list of files written."""
    written = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in LEARNING_FILES:
            path = os.path.join(BASE, name)
            if os.path.isfile(path):
                z.write(path, name)
                written.append(name)
    return written


def import_learning(zip_path):
    """Restore a learning bundle produced by export_learning. Only known
    learning files are extracted (no arbitrary paths). Returns the list of
    files restored."""
    restored = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name in LEARNING_FILES:
                z.extract(name, BASE)
                restored.append(name)
    return restored
