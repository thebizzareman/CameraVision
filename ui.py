"""
On-screen feedback layer: Unicode text, toasts, and voice output.

OpenCV's putText is ASCII-only - it cannot draw "HESAP LÜTFEN". text()
renders through PIL with a real Windows font instead, caching each rendered
string as a bitmap so the per-frame cost is a single alpha blend.

toast() queues transient on-screen notices (database refreshed, sign mode
toggled, snapshot saved...) so events are visible in the window itself, not
only in a console nobody sees when the app starts from run.bat.

speak() voices spelled words and fired gestures through the built-in
Windows speech synthesizer (no extra dependencies; runs detached so the
video loop never blocks).
"""

import os
import subprocess
import time

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
]
_fonts = {}
_text_cache = {}
_CACHE_MAX = 256


def _font(size):
    if size not in _fonts:
        for path in _FONT_CANDIDATES:
            if os.path.isfile(path):
                _fonts[size] = ImageFont.truetype(path, size)
                break
        else:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def _render(s, size, color):
    """RGBA bitmap of the string with a dark outline, cached."""
    key = (s, size, color)
    if key in _text_cache:
        return _text_cache[key]
    font = _font(size)
    pad = max(2, size // 8)
    l, t, r, b = font.getbbox(s)
    img = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - l, pad - t), s, font=font,
              fill=(color[2], color[1], color[0], 255),
              stroke_width=max(1, size // 11), stroke_fill=(0, 0, 0, 255))
    arr = np.array(img)
    if len(_text_cache) >= _CACHE_MAX:
        _text_cache.clear()
    _text_cache[key] = arr
    return arr


def text(frame, s, org, size=22, color=(255, 255, 255)):
    """Draw Unicode text (top-left at org) onto a BGR frame. Returns the
    drawn (width, height). Falls back to cv2.putText without PIL."""
    if not _HAS_PIL:
        cv2.putText(frame, s, (org[0], org[1] + size),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color, 2)
        return (len(s) * size // 2, size)
    bmp = _render(s, size, color)
    h, w = bmp.shape[:2]
    x, y = int(org[0]), int(org[1])
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x < 0 or y < 0 or x2 <= x or y2 <= y:
        return (w, h)
    region = bmp[: y2 - y, : x2 - x]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y:y2, x:x2].astype(np.float32)
    frame[y:y2, x:x2] = (roi * (1 - alpha)
                         + region[:, :, :3].astype(np.float32) * alpha
                         ).astype(np.uint8)
    return (w, h)


# ---------------------------------------------------------------------------
# Toasts
# ---------------------------------------------------------------------------

_toasts = []   # (message, expiry_time)


def toast(msg, seconds=3.5):
    _toasts.append((msg, time.time() + seconds))
    del _toasts[:-4]   # keep the last few


def draw_toasts(frame):
    now = time.time()
    _toasts[:] = [t for t in _toasts if t[1] > now]
    y = frame.shape[0] - 64
    for msg, _ in reversed(_toasts):
        bmp_w, bmp_h = _render(msg, 20, (255, 255, 255)).shape[1::-1] \
            if _HAS_PIL else (len(msg) * 10, 20)
        x = (frame.shape[1] - bmp_w) // 2
        x1, y1 = max(x - 12, 0), y - 6
        x2, y2 = min(x + bmp_w + 12, frame.shape[1]), y + bmp_h + 6
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            cv2.addWeighted(np.zeros_like(roi), 0.6, roi, 0.4, 0, roi)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 160, 60), 1)
        text(frame, msg, (x, y), size=20)
        y -= (bmp_h + 20)


# ---------------------------------------------------------------------------
# Voice (Windows speech synthesizer, detached)
# ---------------------------------------------------------------------------

VOICE = {"on": False}
_last_spoken = {"text": "", "t": 0.0}


def speak(s):
    """Say a short phrase without blocking the video loop."""
    if not VOICE["on"] or not s:
        return
    now = time.time()
    if s == _last_spoken["text"] and now - _last_spoken["t"] < 3.0:
        return   # don't repeat ourselves every frame
    _last_spoken.update(text=s, t=now)
    safe = s.replace("'", "").replace('"', "")[:80]
    cmd = ("Add-Type -AssemblyName System.Speech;"
           "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
           f".Speak('{safe}')")
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
             "-Command", cmd],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
