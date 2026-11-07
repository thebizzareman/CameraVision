# CameraVision

*A personal project by Can Guney (summer 2026), developed with AI-assisted coding using Claude (Anthropic). I designed the features and rules, tested the system live and directed the fixes; Claude helped write the code.*

Live object identification, **hand gesture recognition and ASL
fingerspelling** from your local camera. Runs 100% locally — the pretrained
YOLOX model (OpenCV Zoo) recognizes **80 everyday object classes** (people,
pets, phones, cups, laptops, cars, food, furniture, ...), and MediaPipe hand
tracking (21 landmarks per hand) powers a gesture engine with continuous
per-joint analytics (curl angles 0–1, finger direction vectors).

## Hand gestures

Gesture definitions follow a gesture guide compiled for this project.
Hands are tracked with a full skeleton overlay and labeled live:

- **Per-hand:** Open Palm, Fist, Thumbs Up, Peace (V), Pointing, OK, Rock
- **Two-hand:** **KAPAK** — open palm placed flat on top of a fist
- **Dynamic:** **HESAP LUTFEN** — thumb rubbing across index/middle
  fingertips (the "money" motion; detected by tracking thumb oscillation)
- **Face + hand:** **EYVALLAH** — open palm on the chest below the face;
  **YEMEZLER** — index fingertip pulling the skin under an eye

The Turkish gestures get a big banner when held. Gestures must hold for a
few frames before they fire (temporal smoothing), and every trigger is
logged on-screen with a timestamp. New gestures are one function each in
the `COMPOSITE_GESTURES` / `FACE_HAND_GESTURES` tables in
[gestures.py](gestures.py).

## Facial expressions

The face is tracked with 478 landmarks + 52 muscle "blendshapes"
(MediaPipe face landmarker). Live overlay shows eye, eyebrow and lip
contours, iris positions, and a label with:

- **Expressions:** Eyebrows Raised, Frown, Smile, Mouth Open, Tongue Out,
  Kiss (pucker), Wink
- **Emotions (based on the Facial Action Coding System, FACS):** Happy (genuine
  Duchenne smile), Fake Smile, Surprised, Scared, Angry, Concentration,
  Suspicious, Contempt, Suppressed Laugh — the best-scoring emotion is shown
  in caps next to the face with its confidence
- **Eye tracking:** gaze direction (center / left / right / up / down and
  combinations), read from the iris blendshapes

Expressions feed the same voting + event log as hand gestures.

## Sign language (ASL fingerspelling)

Press **L** to enter sign mode. **All 26 letters of the American manual
alphabet** are recognized: the 24 static letters classified from the
analytics layer (per-joint curl, finger directions, tip geometry, and 3D
thumb depth), plus **J and Z traced in the air** by the motion engine
(their fingertip path is matched against letter templates). Soft template
scoring means every letter gets a 0–1 score and the best fit wins, so an
imperfect hold degrades gracefully instead of misreading — and lookalike
families (A/S/E/T fist shapes, C/O/X curves, U/V/K/R pairs) are separated
by whichever template fits best. All geometry is rotation-invariant, so
tilted and left-handed letters work.

Each letter shows a chip beside the hand with its **confidence score** and,
on the letter being typed, a **hold-progress ring** that fills as the dwell
time completes — so you can see exactly when it will commit. Holding types
the letter into a **spelled-word banner** (release briefly for double
letters; hiding the hand inserts a space; Backspace clears). While sign
mode is on, rule **R14** suppresses all hand gestures — a V-hand is the
letter V, not Peace — but facial expressions stay live.

**It learns your hand.** Press **C** to calibrate (show an open palm, then a
fist) and the extension threshold adapts to your fingers. In sign mode,
press **K** right after a wrong letter and then the intended key — the
classifier nudges that letter's score bias so it reads your hand better
next time. Both persist in `config.json`.

## Dynamic gestures

Beyond static poses, the motion engine ([motion.py](motion.py)) matches
fingertip **trajectories**: the traced letters J and Z, and **Wave** (an
open hand rocking side to side). It shares its oscillation primitive with
the HESAP LÜTFEN money-rub. Motion templates can live in the database as
data, so the research pipeline can grow them too. (Drawing a J or Z no
longer types a stray I/pointing — a hand that's actively tracing suppresses
its static letter.)

## Word signs & sign languages

Fingerspelling spells words letter by letter; real signing is mostly
*whole-word* signs. The word-sign engine ([signword.py](signword.py))
recognizes those as **temporal sequences**: a handshape at a body location,
with a movement or a **motion pattern** (circle / nod / shake), matched by a
per-hand state machine. Built-in ASL signs: **HELLO**, **THANK YOU**,
**PLEASE** (flat hand circling on the chest), **SORRY** (fist circling),
**YES** (nodding fist), **NO** (two fingers shaken). Each sign is pure JSON
— a `sequence` detector with `phases` of `fingers` + `region` + `move` /
`motion` — so the research pipeline can grow the vocabulary with no code.

Signs are **language-tagged**, and you switch languages directly:

- **E** → English / American Sign Language (ASL)
- **T** → Turkish Sign Language (TİD — Türk İşaret Dili)
- **Y** → cycle ASL → TİD → All

Pressing E or T also turns hand detection on, so signs register immediately;
press **L** to fingerspell letters in that language. While fingerspelling,
the closest letter always shows beside your hand — dim as a live guess,
bright once it is confident enough to type — so you always see feedback.

TİD ships with **MERHABA**, **TEŞEKKÜR**, **EVET** (yes) and **HAYIR** (no)
— approximate forms flagged for refinement — and the autonomous researcher
already searches Turkish sources. The active language shows in the HUD; only
that language's word signs fire. (Face tracking moved to **X** and
targets-only to **N** to free E and T for the languages.)

## Gesture database & autonomous online research

Everything the program recognizes is cataloged in
[gesture_db.json](gesture_db.json) — built-in gestures (detected by code),
plus **declarative gestures** described purely as data and detected by
[gesture_db.py](gesture_db.py). The spec schema expresses finger states
(`ext`/`curl`/`half`/`any`), palm orientation, fingertip **spread**,
**crossed** fingers, **thumb-between**-knuckles, **tip-contact**, face
relations, and **two-hand** relations. Facial expressions are data too
(`face_spec`: blendshape thresholds). Adding a gesture or expression means
adding a JSON entry — no code.

[research.py](research.py) grows the database **from the open web, by
itself**:

```
python research.py --auto         # discover sources + merge (autonomous)
python research.py --auto --dry-run   # see what it would learn
python research.py --review        # approve/reject staged proposals
python research.py --url <URL>    # research one specific page
```

In `--auto` mode it searches English **and Turkish** Wikipedia plus arXiv
for gesture/sign/expression articles (no manual URL list), with an on-disk
cache and politeness delays. The generative core is
[prose2spec.py](prose2spec.py): a grammar that parses *any* prose
description ("extending the thumb, index and little finger while the others
are folded") into a detection spec, and FACS/muscle language ("raised
brows, wrinkled nose") into blendshape specs. Every candidate passes a
validation gate — expressibility, duplicate-signature detection (duplicates
become synonyms; strictly-more-specific shapes become overriding gestures),
and synthetic-hand / synthetic-blendshape trigger tests. Merges carry full
provenance (source URL + date); the rest wait in
`research_cache/proposals_pending.json` for `--review`.

The pipeline has autonomously added **ILY** and the **Vulcan Salute** (via
the spread primitive) beyond the seed set. Run research online with **`python
app.py --refresh-db`** — it researches in the background while the camera
works and hot-reloads discoveries live (an on-screen toast announces them).
Or press **R** anytime to reload. The app itself never needs the network.

## Restriction rulebook

All gesture/expression observations pass through a conflict-resolution
filter ([rules.py](rules.py)) before anything can fire. The rules — physical
impossibilities, morphological look-alikes, dynamic-vs-static confusions,
and logical contradictions — are documented with matching IDs in
[RULEBOOK.md](RULEBOOK.md). Suppressions appear in the on-screen event feed
prefixed with `!` so you can see the filter working. To change the rules,
edit RULEBOOK.md and the same-ID rule in rules.py.

## Run it

Double-click `run.bat`, or from a terminal:

```
python app.py
```

A window opens showing your camera with labeled boxes drawn around everything
it recognizes, plus a live "Seen:" summary and FPS counter.

## Look for specific things

Tell it what you care about and those get highlighted in red, with a
FOUND / not-in-view status line:

```
python app.py --targets person,cup,cell phone
```

See every recognizable class name:

```
python app.py --list-classes
```

## Controls (video window focused)

| Key   | Action |
|-------|--------|
| Q/Esc | quit |
| S     | save snapshot to `captures/` |
| D     | toggle object detection on/off |
| G     | toggle hand gesture recognition on/off |
| X     | toggle face expression tracking on/off |
| **E** | **English / American Sign Language (ASL) word signs** |
| **T** | **Turkish Sign Language (TİD) word signs** |
| Y     | cycle word-sign language (ASL → TİD → All) |
| L     | toggle ASL fingerspelling (letter-by-letter) mode |
| Backspace | clear the spelled word (sign mode) |
| C     | calibrate finger thresholds to your hand (open palm, then fist) |
| K     | correct the last letter (sign mode): press K, then the intended key |
| V     | toggle voice output (speaks spelled words / fired gestures) |
| R     | hot-reload the gesture database |
| + / - | raise / lower confidence threshold |
| N     | show target classes only |
| F     | cycle filter: normal → grayscale → edges |

The **web dashboard** (`--web`, `http://127.0.0.1:8765/`) is the richer
interface: it streams the annotated video and adds a live **confidence
sparkline**, a **gesture-history timeline**, a **knowledge-base browser**
(filter by language/type), and a **research-proposal review panel** where
the autonomous researcher's staged candidates are approved or rejected with
one click — approvals merge into the database and hot-reload live. Every
keyboard control is also a button. It costs nothing until a browser
connects (frames are only encoded while someone is watching).

## Portable learning

Everything the program learns — your calibration, letter-correction biases,
and the grown database (including every autonomously-researched gesture) —
travels between machines:

```
python manage.py export mylearning.zip     # on machine A
python manage.py import mylearning.zip      # on machine B, then press R
```

Import only extracts known learning files (no arbitrary paths from the
archive).

## Options

```
python app.py --camera 1        # use a different camera
python app.py --refresh-db      # research online in the background, hot-reload
python app.py --web             # serve the live web dashboard (port 8765)
python research.py --review     # approve/reject staged proposals (CLI)
python manage.py export foo.zip # carry learning + database to another machine
python app.py --record A        # record labeled hands for eval.py, then:
                                #   python eval.py  (per-gesture accuracy table)
python app.py --conf 0.5        # stricter detections
python app.py --no-gestures     # disable hand gestures
python app.py --no-face         # disable face expressions
```

## Requirements

- Python 3 with `opencv-python`, `numpy`, `mediapipe` and `Pillow`:

  ```
  pip install -r requirements.txt
  ```

  Pillow drives the Unicode text overlay so Turkish names like
  **HESAP LÜTFEN** render with correct diacritics; the web dashboard and
  autonomous research use only the standard library.

- Three pretrained models (not included in this repository because of their
  size). Download them into a `models/` folder:

  | File | Purpose | Source |
  |---|---|---|
  | `models/yolox.onnx` | object detection (~36 MB) | YOLOX model from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox) — rename the downloaded `.onnx` file to `yolox.onnx` |
  | `models/hand_landmarker.task` | hand tracking (~8 MB) | [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
  | `models/face_landmarker.task` | face tracking (~4 MB) | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |

## Privacy

All processing happens on your own computer. No camera frames are sent
anywhere and nothing is stored unless you explicitly save a snapshot or a
recording. Your calibration and corrections are kept locally in
`config.json` and `corrections.jsonl`, which are not part of this repository.

## Known limitations

- Letters and gestures are recognised with hand-tuned geometric templates,
  not a trained model; accuracy varies between users (calibration helps).
- The Turkish Sign Language (TİD) signs are approximate and have not been
  validated by a native signer.
