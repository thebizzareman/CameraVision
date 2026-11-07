# Gesture Restriction Rulebook (v2)

Every rule has an ID;
runtime rules are enforced in [rules.py](rules.py) (class `ConflictFilter`),
classifier rules directly in [gestures.py](gestures.py) / [faces.py](faces.py).
All per-frame observations pass through the filter BEFORE temporal voting, and
every suppression is logged on-screen as a `!` event so misfires are visible.

To renew the rulebook: edit this file AND the matching rule in `rules.py`
(they share IDs, so the mapping is one-to-one).

---

## R0. Global temporal confidence (foundation)

No gesture or expression fires from a single frame. Every name must be seen
in >= 4 of the last 7 frames (rolling vote). This is the base defense against
single-frame keypoint noise; the rules below handle *systematic* confusions
that voting alone cannot catch.
*Code: `GestureRecognizer._vote`.*

## R1. Kapak consumes both hands (was 1.1, extended)

Kapak requires two coordinated hands, so while KAPAK is observed the same
actor cannot also produce any face+hand gesture. v2 extension: the original
only excluded Eyvallah; YEMEZLER is excluded for the same physical reason.
*Resolution: KAPAK wins (two-hand coordination is stronger evidence). Drop
EYVALLAH and YEMEZLER.*

## R2. One hand, one body region (was 1.2)

A single hand cannot touch the eye region and the chest at the same moment.
Finger-state requirements already make this near-impossible (Eyvallah needs
an open palm, Yemezler an index-only point), so this rule is a safety net for
keypoint hallucination. Cross-hand (left on chest, right at eye) is
physically possible and stays allowed.
*Resolution: if the SAME hand produces both, keep YEMEZLER (eye contact is
the more specific spatial claim), drop EYVALLAH.*

## R3. Nah vs OK - verify the other three fingers (was 2.1)

Nah: middle/ring/pinky fully folded, thumb protruding between index and
middle knuckles. OK: middle/ring/pinky extended, thumb-index loop.
*Resolution: enforced structurally in the classifier - OK is only emitted
when middle+ring+pinky are extended; Nah only when all four fingers are
curled AND the thumb tip is both laterally close to and beyond the
index/middle knuckle line. One hand can never emit both.*

## R4. Vertical-axis verification for single-digit gestures (was 2.2)

Middle Finger and Thumbs Up both show one extended digit. Identity must come
from joint topology AND axis orientation, not silhouette.
*Resolution (classifier): Thumbs Up requires the thumb tip clearly above the
fist knuckles. v2 extension: Middle Finger now requires the middle finger to
point predominantly upward (tip above MCP, vertical component dominant);
a sideways-extended middle finger is not classified.*

## R5. Dynamic beats static (was 2.3)

While the money-rub (HESAP LUTFEN) motion is being observed, a static OK
detection in the same window is almost certainly a misread rubbing frame.
*Resolution: HESAP LUTFEN suppresses OK for that frame.*

## R6. Emotional contradiction window (was 3.1)

Eyvallah (respect) and Middle Finger (aggression) from the same subject
within ~1.5 s (45 frames) indicates keypoint hallucination.
*Resolution: if both appear in the window, drop BOTH (neither can be
trusted) and log the conflict.*

## R7. Mouth shape exclusivity (v2, new)

A puckered kiss and a wide-open jaw are antagonistic muscle states; both
labels on one frame means the blendshapes disagree.
*Resolution: keep whichever blendshape (mouthPucker vs jawOpen) scores
higher, drop the other.*

## R8. Tongue overrides Kiss (v2, new)

A protruding tongue pushes the lips into a shape that can score as a pucker.
*Resolution: Tongue Out suppresses Kiss.*

## R9. Smile/Frown co-activation (v2, new)

Squinting and smiling mechanically co-activate the brow-down muscles
(observed 0.83 browDown on a clearly smiling test portrait).
*Resolution: enforced in the expression rules - Frown additionally requires
the smile muscles to be quiet (< 0.3). Code: `faces.EXPRESSION_RULES`.*

---

# Facial emotion rules (v3, based on FACS)

Blendshapes act as Action Unit proxies (AU1=browInnerUp, AU2=browOuterUp,
AU4=browDown, AU5=eyeWide, AU6=cheekSquint, AU7=eyeSquint, AU12=mouthSmile,
AU14=mouthDimple, AU15=mouthFrown, AU20=mouthStretch, AU24=mouthPress,
AU26=jawOpen). MediaPipe computes blendshapes on the pose-normalized mesh,
which satisfies the head-pose normalization requirement (doc section 4.3).
*Code: `faces.EMOTION_RULES` / `faces.pick_emotion`.*

## R10. Duchenne verification (doc 2.1)

A smile (AU12) only counts as HAPPY when the cheek raiser (AU6) is also
active. AU12 without AU6 is reported as FAKE SMILE instead. Both require
near-symmetric smiles (else R11 applies).

## R11. Asymmetry classifiers (doc 2.2, 2.3, 4.1)

Left/right activation is compared independently. A one-sided outer-brow
raise with narrowed eyes is SUSPICIOUS; a one-sided lip pull/dimple is
CONTEMPT (the only strictly unilateral emotion). Symmetric activations can
never produce these labels.

## R12. Frame-level emotion exclusivity (doc 4.2)

Emotion scores compete; only the single best-scoring emotion above the
minimum score is reported per frame, then passes the same 4-of-7 temporal
vote as everything else (micro-expression transition frames cannot fire).

## R13. Angry vs Concentration mouth differentiator (doc 3.3)

Both share AU4+AU7 (furrowed brow, tightened lids). The mouth decides:
pressed/depressed lips -> ANGRY; neutral mouth -> CONCENTRATION.

## R14. Sign mode exclusivity (classifier-level, gestures.py)

While ASL sign mode is on (L key), the hands belong to the fingerspelling
layer. Every hand-gesture observation - single-hand, two-hand, dynamic and
face+hand - is suppressed before the conflict filter, because most manual
alphabet letters are morphologically identical to gestures (V = Peace,
Y = Shaka, T = Nah/fig, F = OK, A = Thumbs Up topology, B = Open Palm...).
Reporting both readings for one hand shape would be a contradiction by
construction. Facial expressions and emotions stay live: the face is not
involved in fingerspelling.

Letter classification runs on the continuous analytics layer (per-joint
curl 0..1 from bend angles, finger direction vectors) rather than the
binary extended/curled states, because letter pairs like U/R (fingers
together vs crossed), C/O (open arc vs closed circle) and X/D (half-hook
vs straight index) differ only at that finer resolution. J and Z require
tracing motion and are recognised separately by the motion engine
(`motion.py`).

## Emotion summary

| Emotion | AU combination | Key gate |
|---------|----------------|----------|
| Happy | AU12 + AU6 | cheek raiser active, symmetric |
| Fake Smile | AU12 alone | cheek raiser quiet, symmetric |
| Surprised | AU1+AU2+AU5+AU26 | no lip stretch |
| Scared | AU1+AU5+AU20/AU15 | no smile |
| Angry | AU4+AU7+AU24/AU15 | mouth engaged, no smile |
| Concentration | AU4+AU7 | mouth neutral |
| Suspicious | unilateral AU2 + AU7 | brow asymmetry > 0.25 |
| Contempt | unilateral AU12/AU14 | smile asymmetry > 0.25 |
| Suppressed Laugh | AU12 vs AU24 | press and smile both active |

---

## Conflict Resolution Matrix (v2)

| ID | Gesture 1 | Gesture 2 | Conflict type | Winner |
|----|-----------|-----------|---------------|--------|
| R1 | KAPAK | EYVALLAH, YEMEZLER | physical (arm span) | KAPAK |
| R2 | YEMEZLER | EYVALLAH (same hand) | spatial | YEMEZLER |
| R3 | Nah | OK | morphological | structurally exclusive |
| R4 | Middle Finger | Thumbs Up / sideways digit | axis | vertical topology only |
| R5 | HESAP LUTFEN | OK | dynamic vs static | HESAP LUTFEN |
| R6 | EYVALLAH | Middle Finger | logical/emotional | neither (both dropped) |
| R7 | Kiss | Mouth Open | facial-muscle | higher blendshape |
| R8 | Tongue Out | Kiss | facial-muscle | Tongue Out |
| R9 | Smile | Frown | facial-muscle | Smile |
| R14 | ASL letters (sign mode) | all hand gestures | morphological (letter = gesture shape) | letters |

## R15. Per-hand precision arbitration (gestures.py `_arbitrate`)

Each hand does ONE thing at a time. Before the conflict filter, every
observation is tagged with a specificity tier and, per hand, only the most
specific survives - so a sign no longer stacks with the generic pose it is
built from. Tiers, most specific first:

| Tier | Kind | Example |
|------|------|---------|
| 0 | word sign (handshape + location + motion) | HELLO, PLEASE, YES |
| 1 | two-hand / face+hand composite | KAPAK, EYVALLAH |
| 2 | dynamic motion | HESAP LÜTFEN, Wave |
| 3 | declarative single-hand handshape | Shaka, ILY, Vulcan |
| 4 | builtin static pose | Open Palm, Peace, Fist |

A two-hand sign (tier 1, both hands) suppresses both hands' single-hand
reads. Facial expressions and emotions are a separate modality and always
pass (a face can emote while a hand signs). The hand label names only
hand_label poses; card gestures are named on their card, never twice. This
runs BEFORE R1-R14, which then resolve the remaining cross-hand conflicts.

## Changelog

- **v7 (2026-07-07):** precision arbitration (R15) - one gesture per hand,
  most specific wins, cards stop stacking. New motion-pattern word signs
  (PLEASE, SORRY circle; YES nod; NO shake) via a circle/nod/shake
  classifier. Direct language keys: E = English (ASL), T = Turkish (TİD);
  face moved to X, targets-only to N. TİD gained EVET/HAYIR.
- **v6 (2026-07-07):** word-sign layer (`signword.py`) - whole signs as
  temporal handshape+location+movement sequences (HELLO, THANK YOU), a
  `sequence` detector kind in the database. Sign languages are tagged and
  filterable (ASL / TİD); Turkish Sign Language seeded (MERHABA, TEŞEKKÜR).
  The web dashboard became the primary UI (confidence sparkline, gesture
  timeline, knowledge-base browser, one-click research-proposal review that
  merges + hot-reloads). Portable learning: `manage.py export/import`
  carries calibration, corrections and the grown database between machines.
  Bug fix: tracing J/Z no longer commits a spurious static letter first.
- **v5 (2026-07-06):** program-wide quality pass. Generative research: a
  prose-to-spec grammar (`prose2spec.py`) parses arbitrary gesture and
  FACS/expression descriptions, and `research.py --auto` discovers its own
  sources (English + Turkish Wikipedia, arXiv) — autonomously added ILY and
  the Vulcan Salute. Data-driven facial expressions (`face_spec`) and
  two-hand specs; new geometric primitives (spread, crossed, thumb-between,
  tip-contact, palm orientation). 3D depth (thumb front/behind) sharpens the
  A/S/E/T fist family. Motion engine (`motion.py`): traced letters J/Z and a
  Wave gesture. Per-user calibration (C) and a wrong-letter correction loop
  (K) that learns letter biases. Feedback layer: Unicode text (HESAP LÜTFEN
  renders with its diacritics), on-screen toasts, letter confidence + a
  hold-progress ring, voice output (V), and a browser dashboard (`--web`).
  Offline accuracy harness (`--record` + `eval.py`).
- **v4.1 (2026-07-05):** letter classifier rebuilt as soft template
  scoring after live testing - every letter scores 0..1 and the best fit
  wins, so an imperfect hold degrades to a lower score instead of
  "undetected", and lookalikes (A/S/E/T fist family, C/O/X curve family,
  U/V/K/R pair family) are separated by whichever template fits best. All
  positional tests moved into hand-axis coordinates, fixing tilted and
  left-hand letters. Per-hand letters get a 5-frame majority vote (chip
  flicker), and with two hands visible the speller sticks to the hand it
  is already following instead of whichever the tracker lists first.
- **v4 (2026-07-05):** ASL fingerspelling layer (signs.py): 24 static
  letters + speller with commit-after-hold; R14 sign-mode exclusivity.
  Analytic precision upgrade: continuous per-joint curl (bend angles),
  finger direction vectors, angle-gated extension tests, and a new "half"
  finger state in the declarative spec schema.
- **v3 (2026-07-05):** FACS emotion layer
  (R10-R13): Duchenne genuine/fake smile check, asymmetry-based Suspicion
  and Contempt, top-1 emotion exclusivity, Angry/Concentration mouth
  differentiator. Display: basic hand poses (Open Palm, Fist, Rock) are
  labeled at the hand instead of raising pop-up cards.
- **v2 (2026-07-04):** rule IDs; R1 extended to Yemezler; R2 winner defined;
  R4 verticality made mandatory for Middle Finger; R6 drops both sides;
  new face rules R7-R9; R0 documented; suppressions logged to the on-screen
  event feed.
- **v1:** original rulebook.
