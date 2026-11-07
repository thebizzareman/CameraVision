"""
CameraVision - Live object identification from your local camera.

Uses OpenCV's DNN module with a pretrained YOLOX model (80 COCO classes).
Everything runs locally; no cloud, no internet needed after setup.

Hand gestures (gestures.py, MediaPipe hand tracking; definitions follow
the el_hareketleri_kilavuzu gesture dictionary):
  Per-hand:   Open Palm, Fist, Thumbs Up, Peace (V), Pointing, OK, Rock,
              Nah (fig sign), Middle Finger
  Two-hand:   KAPAK - open palm placed on top of a fist
  Dynamic:    HESAP LUTFEN - thumb rubbing index/middle tips (money sign)
  Face+hand:  EYVALLAH - open palm on the chest below the face
              YEMEZLER - index fingertip pulling under an eye

Facial expressions (faces.py, MediaPipe face landmarker blendshapes):
  Eyebrows Raised, Frown, Smile, Mouth Open, Tongue Out, Kiss, Wink,
  plus live gaze direction (eye tracking).

Sign language (signs.py): the full ASL manual alphabet - 24 static letters
scored from per-joint curl analytics + 3D thumb depth, and J/Z traced in
the air (motion.py trajectory templates) - with a spelled-word buffer,
per-letter confidence and a hold-progress ring. Toggle with L.

Controls (with the video window focused):
  Q / Esc   quit
  S         save a snapshot (with detections drawn) to captures/
  D         toggle object detection on/off
  G         toggle hand gesture recognition on/off
  X         toggle face expression tracking on/off
  E         English / American Sign Language (ASL) word signs
  T         Turkish Sign Language (TID) word signs
  Y         cycle word-sign language: ASL -> TID -> All
  L         toggle ASL fingerspelling (letter-by-letter spelling) mode
  Backspace clear the spelled word (sign mode)
  C         calibrate finger thresholds to your hand (open palm, then fist)
  K         correct the last letter (sign mode): press K, then the intended key
  V         toggle voice output (speaks spelled words / fired gestures)
  R         hot-reload the gesture database
  + / -     raise / lower the confidence threshold
  N         toggle "targets only" mode (only show classes passed via --targets)
  F         cycle view filter: normal -> grayscale -> edges

Flags: --refresh-db (background online research + live hot-reload),
--web (browser dashboard on 127.0.0.1:8765), --record LABEL (log labeled
hands for eval.py), --targets, --camera, --conf, --no-gestures, --no-face.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolox.onnx")
CAPTURE_DIR = os.path.join(BASE_DIR, "captures")
INPUT_SIZE = 640

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Stable, distinct color per class (BGR)
rng = np.random.default_rng(7)
CLASS_COLORS = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in COCO_CLASSES]


class YoloXDetector:
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            sys.exit(f"Model file not found: {model_path}\nRe-run setup or re-download yolox.onnx.")
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.grids, self.strides = self._build_grids()

    @staticmethod
    def _build_grids():
        grids, strides = [], []
        for s in (8, 16, 32):
            n = INPUT_SIZE // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], 1), s))
        return (np.concatenate(grids).astype(np.float32),
                np.concatenate(strides).astype(np.float32))

    def detect(self, frame, conf_threshold=0.35, nms_threshold=0.45):
        """Returns a list of (class_id, confidence, (x, y, w, h)) in frame coordinates."""
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized

        # YOLOX expects raw 0-255 pixel values, no normalization
        blob = cv2.dnn.blobFromImage(padded)
        self.net.setInput(blob)
        out = self.net.forward()[0]  # (8400, 85)

        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides

        scores = out[:, 4:5] * out[:, 5:]  # objectness * class prob
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(class_ids)), class_ids]

        keep = confidences >= conf_threshold
        if not np.any(keep):
            return []

        boxes_cxcywh = out[keep, :4] / scale
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y
        boxes[:, 2] = boxes_cxcywh[:, 2]  # w
        boxes[:, 3] = boxes_cxcywh[:, 3]  # h

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), confidences.tolist(), conf_threshold, nms_threshold
        )
        results = []
        for i in np.array(indices).flatten():
            x, y, bw, bh = boxes[i]
            x = max(0, int(x)); y = max(0, int(y))
            bw = min(int(bw), w - x); bh = min(int(bh), h - y)
            results.append((int(class_ids[i]), float(confidences[i]), (x, y, bw, bh)))
        return results


def draw_detections(frame, detections, targets=None, targets_only=False):
    counts = {}
    for class_id, conf, (x, y, w, h) in detections:
        name = COCO_CLASSES[class_id]
        is_target = targets is not None and name in targets
        if targets_only and targets and not is_target:
            continue
        counts[name] = counts.get(name, 0) + 1

        color = (0, 0, 255) if is_target else CLASS_COLORS[class_id]
        thickness = 3 if is_target else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y - 8 if y - 8 > th else y + h + th + 8
        cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 6, ty + 4), color, -1)
        cv2.putText(frame, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    return counts


def draw_hud(frame, fps, conf, detecting, counts, targets, targets_only,
             filter_name, gesturing, active_gestures, facing=False,
             signing=False, sign_lang=None):
    lang = sign_lang or "All"
    lines = [f"FPS {fps:.1f} | conf {conf:.2f} | objects {'ON' if detecting else 'OFF'}"
             + f" | gestures {'ON' if gesturing else 'OFF'}"
             + f" | face {'ON' if facing else 'OFF'}"
             + (" | ASL SIGN MODE" if signing else "")
             + (f" | signs: {lang}" if gesturing and not signing else "")
             + (f" | filter {filter_name}" if filter_name != "normal" else "")]
    if counts:
        lines.append("Seen: " + ", ".join(f"{v}x {k}" for k, v in sorted(counts.items())))
    if targets:
        found = [t for t in targets if t in counts]
        status = f"TARGETS {'(only)' if targets_only else ''}: {', '.join(targets)}"
        status += f"  ->  FOUND: {', '.join(found)}" if found else "  ->  not in view"
        lines.append(status)
    if active_gestures:
        lines.append("Gestures: " + ", ".join(sorted(active_gestures)))

    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        y += 30
    return y


def apply_filter(frame, mode):
    if mode == "gray":
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if mode == "edges":
        return cv2.cvtColor(cv2.Canny(frame, 80, 160), cv2.COLOR_GRAY2BGR)
    return frame


def open_camera(index):
    # CAP_DSHOW opens much faster on Windows; fall back to default backend
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Live object identification from your camera.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--targets", type=str, default="",
                        help="comma-separated class names to highlight, e.g. person,dog,cup")
    parser.add_argument("--list-classes", action="store_true",
                        help="print the 80 recognizable object classes and exit")
    parser.add_argument("--no-gestures", action="store_true",
                        help="start with hand gesture recognition disabled")
    parser.add_argument("--no-face", action="store_true",
                        help="start with face expression tracking disabled")
    parser.add_argument("--refresh-db", action="store_true",
                        help="research online sources in the background at "
                             "startup and hot-reload anything discovered")
    parser.add_argument("--web", action="store_true",
                        help="serve a live web dashboard on 127.0.0.1:8765 "
                             "(stream + controls in the browser)")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--record", metavar="LABEL",
                        help="record labeled hand landmarks to recordings/ "
                             "for offline accuracy evaluation (eval.py)")
    args = parser.parse_args()

    if args.list_classes:
        print("\n".join(COCO_CLASSES))
        return

    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    unknown = [t for t in targets if t not in COCO_CLASSES]
    if unknown:
        sys.exit(f"Unknown target class(es): {', '.join(unknown)}\n"
                 f"Run with --list-classes to see valid names.")

    print("Loading models...")
    detector = YoloXDetector(MODEL_PATH)

    recognizer = None
    if not args.no_gestures:
        try:
            import gestures
            recognizer = gestures.GestureRecognizer()
        except ImportError as exc:
            print(f"Gesture recognition unavailable ({exc}); "
                  f"run: pip install mediapipe")

    face_analyzer = None
    if not args.no_face:
        try:
            import faces
            face_analyzer = faces.FaceAnalyzer()
        except ImportError as exc:
            print(f"Face expression tracking unavailable ({exc})")

    # Autonomous online refresh: research runs in the background while the
    # camera works; discoveries hot-reload without touching the video loop.
    refresh = {"msg": None}
    if args.refresh_db and recognizer is not None:
        import threading

        def _bg_refresh():
            try:
                import research
                s = research.run(auto=True)
                refresh["msg"] = (f"DB refreshed: +{s['merged']} new, "
                                  f"{s['synonyms']} synonym(s), "
                                  f"{s['pending']} pending")
            except Exception as exc:
                refresh["msg"] = f"DB refresh failed: {exc}"

        threading.Thread(target=_bg_refresh, daemon=True).start()
        print("Online research running in the background (--refresh-db)...")

    web_srv = None
    if args.web:
        import web
        import research as _research

        def _db_rows():
            import gesture_db
            return [{"name": e["name"], "type": e.get("type", ""),
                     "language": e.get("language", ""),
                     "origin": e.get("origin", "")}
                    for e in gesture_db.load()
                    if not e["name"].startswith("_")]

        def _pending():
            return [{"name": p["name"], "reason": p.get("reason", "")}
                    for p in _research.list_pending()]

        def _resolve(name, action):
            res = _research.resolve_pending(name, action)
            if res.get("ok") and recognizer is not None:
                recognizer.reload_db()
            return res

        web_srv = web.start(args.web_port, db=_db_rows,
                            pending=_pending, resolve=_resolve)
        print(f"Web dashboard: http://127.0.0.1:{args.web_port}/")

    record_file = None
    if args.record:
        import json as _json
        os.makedirs(os.path.join(BASE_DIR, "recordings"), exist_ok=True)
        rec_path = os.path.join(
            BASE_DIR, "recordings",
            f"{args.record}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl")
        record_file = open(rec_path, "w", encoding="utf-8")
        print(f"Recording '{args.record}' hands to {rec_path}")

    print(f"Opening camera {args.camera}...")
    cap = open_camera(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. "
                 f"Check that a camera is connected and not in use by another app, "
                 f"or try --camera 1.")

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    conf = args.conf
    # calibration + correction state (C / K keys)
    calib = {"phase": None, "curls": []}
    correcting = False
    prev_word = ""
    prev_event_count = 0
    import ui
    import config as user_config
    ui.VOICE["on"] = bool(user_config.load().get("voice", False))
    detecting = True
    gesturing = recognizer is not None
    facing = face_analyzer is not None
    targets_only = False
    filters = ["normal", "gray", "edges"]
    filter_idx = 0
    fps = 0.0
    frame_no = 0
    cached_detections = []
    window = ("CameraVision - Q quit | D objects | G gestures | X face | "
              "E English SL | T Turkish SL | L spell | C calibrate | V voice")
    print("Running. Press Q in the video window to quit.")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed; stopping.")
            break
        frame_no += 1

        display = apply_filter(frame, filters[filter_idx])

        counts = {}
        if detecting:
            # Objects change slowly; the hand and face trackers need every
            # frame, so YOLO runs on a stride when trackers are active.
            stride = 1 + int(gesturing) + int(facing)
            if frame_no % stride == 1 or stride == 1:
                cached_detections = detector.detect(frame, conf_threshold=conf)
            counts = draw_detections(display, cached_detections, targets,
                                     targets_only)

        face = None
        if facing and face_analyzer is not None:
            import faces
            face = face_analyzer.process(frame)
            if face is not None:
                faces.draw_face(display, face)

        # background research finished -> hot-reload + on-screen notice
        if refresh["msg"] and recognizer is not None:
            msg = refresh["msg"]
            refresh["msg"] = None
            n = recognizer.reload_db()
            print(f"{msg} | {n} data-driven detector(s) active")
            recognizer._log_event(msg)
            prev_event_count = recognizer.event_count
            ui.toast(msg)

        hands, active_gestures = [], set()
        if gesturing and recognizer is not None:
            import gestures
            import ui
            hands, active_gestures = recognizer.process(frame, face)
            if record_file is not None and hands:
                import json as _json
                h0 = hands[0]
                record_file.write(_json.dumps({
                    "lms": [[float(x), float(y), float(z)] for (x, y), z
                            in zip(h0.points / [frame.shape[1],
                                                frame.shape[0]], h0.z)],
                    "handedness": h0.handedness}) + "\n")
            gestures.draw_hands(display, hands, active_gestures)
            if recognizer.sign_mode:
                import signs
                signs.draw_letters(display, hands,
                                   current=recognizer.speller._current,
                                   progress=recognizer.speller.progress)
                signs.draw_spelling(display, recognizer.speller.word)
                # voice: speak each word as its boundary space commits
                word = recognizer.speller.word
                if (word.endswith(" ") and word != prev_word
                        and word.strip()):
                    ui.speak(word.split()[-1])
                prev_word = word
            # voice: speak newly fired gestures (skip suppression notes)
            new_k = recognizer.event_count - prev_event_count
            if new_k:
                for _, ev in recognizer.events[-new_k:]:
                    if not ev.startswith("!") and not ev.startswith("DB "):
                        ui.speak(gestures.DISPLAY_NAMES.get(ev, ev))
                prev_event_count = recognizer.event_count

        # calibration: collect curls from the visible hand, phase by phase
        if calib["phase"] and hands:
            calib["curls"].append(list(hands[0].finger_curls))
            if len(calib["curls"]) >= 40:
                mean = float(np.mean(calib["curls"]))
                if calib["phase"] == "open":
                    calib.update(phase="fist", curls=[], open_mean=mean)
                else:
                    import config as user_config
                    import gestures as g
                    t = user_config.calibrate_threshold(
                        calib.get("open_mean", 0.1), mean)
                    g.USER["curl_ext_max"] = t
                    cfg = user_config.load()
                    cfg["curl_ext_max"] = t
                    user_config.save(cfg)
                    calib.update(phase=None, curls=[])
                    print(f"Calibrated: extension threshold {t:.2f}")
                    ui.toast(f"Calibrated to your hand (threshold {t:.2f})")
        if calib["phase"]:
            prompt = ("CALIBRATION: show an OPEN PALM"
                      if calib["phase"] == "open"
                      else "CALIBRATION: now make a FIST")
            cv2.putText(display, prompt,
                        (40, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(display, prompt,
                        (40, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 220, 255), 2)
        if correcting:
            cv2.putText(display, "WRONG LETTER: press the intended key (A-Y)",
                        (40, display.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(display, "WRONG LETTER: press the intended key (A-Y)",
                        (40, display.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 220, 255), 2)

        hud_bottom = draw_hud(display, fps, conf, detecting, counts, targets,
                              targets_only, filters[filter_idx], gesturing,
                              active_gestures, facing,
                              signing=(recognizer is not None
                                       and recognizer.sign_mode),
                              sign_lang=(recognizer.sign_language
                                         if recognizer is not None else None))
        if gesturing and recognizer is not None and recognizer.events:
            import gestures
            gestures.draw_events(display, recognizer.events, hud_bottom + 8)
        ui.draw_toasts(display)
        cv2.imshow(window, display)
        if web_srv is not None:
            import web
            lang = (recognizer.sign_language or "All"
                    if recognizer is not None else "")
            modes = (f"objects {'on' if detecting else 'off'}, "
                     f"gestures {'on' if gesturing else 'off'}, "
                     f"face {'on' if facing else 'off'}"
                     + (", SIGN" if recognizer is not None
                        and recognizer.sign_mode else f", signs:{lang}")
                     + (", voice" if ui.VOICE["on"] else ""))
            letter_score = 0.0
            if hands and recognizer is not None and recognizer.sign_mode:
                letter_score = max((float(getattr(h, "sign_score", 0.0))
                                    for h in hands), default=0.0)
            web.publish(display, {
                "fps": round(fps, 1), "conf": round(conf, 2),
                "word": (recognizer.speller.word
                         if recognizer is not None else ""),
                "gestures": sorted(active_gestures), "modes": modes,
                "letter_score": round(letter_score, 2),
                "history": (recognizer.events
                            if recognizer is not None else [])})
        fps = 0.9 * fps + 0.1 * (1.0 / max(time.perf_counter() - t0, 1e-6))

        key = cv2.waitKey(1) & 0xFF
        # merge clicks from the web dashboard into the same key handling
        if key == 255 and web_srv is not None:
            import web
            clicked = web.pop_keys()
            if clicked:
                key = ord(clicked[0])
        # an armed correction consumes the next letter key (A-Y)
        if correcting and ord("a") <= key <= ord("z"):
            correcting = False
            intended = chr(key).upper()
            shown = recognizer.speller._current if recognizer else ""
            import config as user_config
            import json as _json
            cfg = user_config.load()
            user_config.apply_correction(cfg, shown, intended)
            user_config.save(cfg)
            import signs as _signs
            _signs.set_bias(cfg["letter_bias"])
            with open(os.path.join(BASE_DIR, "corrections.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(_json.dumps({
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "shown": shown, "intended": intended}) + "\n")
            print(f"Correction learned: {shown or '?'} -> {intended}")
            ui.toast(f"Correction learned: {shown or '?'} -> {intended}")
            continue
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            path = os.path.join(CAPTURE_DIR,
                                datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg"))
            cv2.imwrite(path, display)
            print(f"Saved {path}")
            ui.toast(f"Snapshot saved: {os.path.basename(path)}")
        elif key == ord("d"):
            detecting = not detecting
        elif key == ord("g") and recognizer is not None:
            gesturing = not gesturing
        elif key == ord("x") and face_analyzer is not None:
            facing = not facing            # face toggle (moved off E)
            ui.toast(f"Face tracking {'ON' if facing else 'OFF'}")
        elif key == ord("e") and recognizer is not None:
            recognizer.set_language("ASL")          # English / American SL
            gesturing = True                        # so signs are detected
            print("Sign language: English (ASL)")
            ui.toast("English Sign Language ON - sign HELLO / PLEASE / YES, "
                     "or press L to fingerspell letters")
        elif key == ord("t") and recognizer is not None:
            recognizer.set_language("TID")          # Turkish SL (TID)
            gesturing = True
            print("Sign language: Turkish (TID)")
            ui.toast("Turkish Sign Language (TID) ON - sign MERHABA / EVET, "
                     "or press L to fingerspell letters")
        elif key == ord("r") and recognizer is not None:
            n = recognizer.reload_db()
            print(f"Reloaded gesture database: {n} spec gesture(s) active")
            ui.toast(f"Gesture DB reloaded: {n} data-driven detector(s)")
        elif key == ord("l") and recognizer is not None:
            recognizer.sign_mode = not recognizer.sign_mode
            recognizer.speller.clear()
            state = "ON" if recognizer.sign_mode else "OFF"
            print(f"ASL sign mode {state}")
            ui.toast(f"ASL sign mode {state}")
        elif key == 8 and recognizer is not None:  # Backspace
            recognizer.speller.clear()
        elif key == ord("c") and recognizer is not None:
            gesturing = True   # calibration needs live hand tracking
            calib.update(phase="open", curls=[])
            print("Calibration started: show an open palm...")
        elif (key == ord("k") and recognizer is not None
              and recognizer.sign_mode):
            correcting = True
        elif key == ord("y") and recognizer is not None:
            lang = recognizer.cycle_language()
            print(f"Sign language: {lang}")
            ui.toast(f"Sign language: {lang} word signs")
        elif key == ord("v"):
            ui.VOICE["on"] = not ui.VOICE["on"]
            cfg = user_config.load()
            cfg["voice"] = ui.VOICE["on"]
            user_config.save(cfg)
            ui.toast(f"Voice {'ON' if ui.VOICE['on'] else 'OFF'}")
            if ui.VOICE["on"]:
                ui.speak("Voice on")
        elif key == ord("n"):
            targets_only = not targets_only    # targets-only (moved off T)
            ui.toast(f"Targets-only {'ON' if targets_only else 'OFF'}")
        elif key == ord("f"):
            filter_idx = (filter_idx + 1) % len(filters)
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)

        # Window closed with the X button
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if record_file is not None:
        record_file.close()
    if recognizer is not None:
        recognizer.close()
    if face_analyzer is not None:
        face_analyzer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
