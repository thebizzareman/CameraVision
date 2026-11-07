"""
Generative prose-to-spec grammar.

This is what makes the research pipeline generative instead of curated:
rather than one handwritten regex per known gesture (research.py's original
RULE_LIBRARY), this module parses ANY prose description of a hand gesture or
facial expression - from Wikipedia articles, research-paper abstracts, sign
language references - and composes a declarative detection spec from grammar
rules over the vocabulary people actually use:

  hand gestures:  finger names ("thumb", "index and middle fingers",
                  "the little finger"), finger groups ("all four fingers",
                  "the other fingers", "the middle three fingers"), state
                  verbs ("extended", "curled into the palm", "half bent"),
                  whole-hand bases ("a fist", "an open palm"), and body
                  relations ("raised to the forehead", "on the chest")

  expressions:    FACS Action Units ("AU12", "AU4+AU7") and muscle phrases
                  ("raised eyebrows", "lip corner puller", "pressed lips",
                  "jaw drop"), mapped onto MediaPipe blendshape thresholds

Every extraction carries the exact evidence sentence and a confidence score.
Nothing here touches the database - research.py runs each proposal through
the validation gate (expressibility, duplicate detection, synthetic tests)
before anything merges.
"""

import re

# ---------------------------------------------------------------------------
# Hand-gesture grammar
# ---------------------------------------------------------------------------

FINGER_ALIASES = [
    (r"forefingers?|first fingers?|index fingers?|index digits?|index",
     "index"),
    (r"middle fingers?|second fingers?|long fingers?|middle", "middle"),
    (r"ring fingers?|third fingers?|ring", "ring"),
    (r"little fingers?|little digits?|pinkies|pinkie|pinky fingers?|pinky"
     r"|fifth fingers?", "pinky"),
    (r"thumbs?", "thumb"),
]
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
NON_THUMB = ("index", "middle", "ring", "pinky")

# State cue vocabulary. Order matters: half-cues are checked before curl/ext
# so "partially bent" doesn't read as plain "bent".
HALF_CUES = (r"partial(?:ly)?\s+(?:bent|curled|closed|extended|flexed)"
             r"|half[- ]?(?:bent|curled|closed|open)|hooked|crooked")
EXT_CUES = (r"extend(?:ed|ing|s)?|outstretch(?:ed)?|straight(?:ened)?|erect"
            r"|upright|rais(?:ed|ing)|stretch(?:ed|ing)?|stick(?:ing|s)?\s+"
            r"(?:up|out)|held\s+(?:up|out)|point(?:ed|ing|s)?(?:\s+(?:up|out"
            r"|forward|upward|outward))?|protrud(?:ed|ing|es)|splay(?:ed)?"
            r"|open(?:ed)?|up(?:ward)?")
CURL_CUES = (r"curl(?:ed|ing|s)?|fold(?:ed|ing|s)?|clench(?:ed|ing)?"
             r"|clos(?:ed|ing)|tuck(?:ed|ing)?|flex(?:ed|ing)?|bent|bend(?:ing)?"
             r"|drawn\s+in(?:to)?|against\s+the\s+palm|into\s+the\s+palm"
             r"|held\s+down(?:\s+by)?|hold(?:ing|s)?\s+down|press(?:ed|ing)?\s+"
             r"(?:down|into|against)|down")

GROUP_PATTERNS = [
    # ordered most-specific first. "three middle fingers" needs the explicit
    # "three": a bare "middle fingers" inside an enumeration ("the thumb,
    # index, and middle fingers") is just the middle finger, plural.
    (r"(?:the\s+)?three\s+middle\s+fingers|(?:the\s+)?middle\s+three\s+fingers",
     ("index", "middle", "ring")),
    (r"(?:the\s+)?first\s+two\s+fingers", ("index", "middle")),
    (r"(?:the\s+)?first\s+three\s+fingers", ("index", "middle", "ring")),
    (r"all\s+(?:four\s+|the\s+)?fingers(?:\s+and\s+(?:the\s+)?thumb)?",
     NON_THUMB),
    (r"(?:the\s+)?(?:other|remaining)\s+(?:fingers|digits)"
     r"|(?:the\s+)?rest(?:\s+of\s+the\s+(?:fingers|hand|digits))?",
     "COMPLEMENT"),
]

FIST_CUE = re.compile(r"\b(?:a\s+|the\s+|closed\s+)?fist\b|closed\s+hand")
OPEN_CUE = re.compile(r"open\s+(?:palm|hand)|flat\s+(?:hand|palm)"
                      r"|palm\s+(?:held\s+)?open|all\s+fingers\s+"
                      r"(?:extended|spread|open)")

FACE_REGIONS = [
    # "heart" is deliberately absent: prose like "form a small heart"
    # describes a SHAPE, not the hand touching the chest
    (r"forehead|brow(?!\w)|temple", "forehead", 1.3),
    (r"chest|breast", "chest", 1.6),
    (r"chin|cheek|lips|mouth|nose|eye|ear|face|head", "face", 1.4),
]

# a sentence must look like it DESCRIBES a gesture, not merely mention hands
GESTURE_CONTEXT = re.compile(
    r"\b(?:sign|gesture|salute|greeting|mudra|letter)\b"
    r"|\bmade\s+by\b|\bformed\s+by\b|\bperformed\s+by\b|\bconsists?\s+of\b"
    r"|\bexecuted\s+by\b|\bdisplayed\s+by\b")

NAME_STOPWORDS = {
    "the", "this", "that", "it", "a", "an", "one", "another", "such",
    "common", "popular", "similar", "various", "other", "using", "made",
    "hand", "finger", "fingers", "gesture", "gestures", "sign", "signs",
    "in", "on", "when", "where", "while", "many", "some", "most", "all",
}

# names made only of these are descriptions, not gesture names
GENERIC_NAME_WORDS = NAME_STOPWORDS | {
    "signal", "signals", "list", "number", "numbers", "letter", "letters",
    "counting", "greeting", "greetings", "chinese", "american", "turkish",
    "arabic", "indian", "european", "western", "eastern", "manual",
    "alphabet", "language", "people", "person", "culture", "cultures",
    "several", "certain", "these", "those", "both", "mudra", "mudras",
}


def _plausible_name(name):
    words = name.split()
    if not words or len(words) > 4:
        return False
    if all(w.lower() in GENERIC_NAME_WORDS for w in words):
        return False
    if name.lower().startswith(("list of", "lists of")):
        return False
    return True


def _major_clauses(sentence):
    """Contrast boundaries: the running state cue resets across these."""
    return re.split(r";|\bwhile\b|\bwhereas\b|\bbut\b", sentence)


def _fingers_in(clause):
    """Fingers/groups mentioned in a clause, in order of appearance."""
    found = []
    spans = []
    for pat, fingers in GROUP_PATTERNS:
        for m in re.finditer(pat, clause):
            found.append((m.start(), fingers))
            spans.append((m.start(), m.end()))
    for pat, name in FINGER_ALIASES:
        for m in re.finditer(r"\b(?:%s)\b" % pat, clause):
            # skip mentions inside an already-matched group span
            if any(a <= m.start() < b for a, b in spans):
                continue
            found.append((m.start(), (name,)))
            spans.append((m.start(), m.end()))
    found.sort(key=lambda t: t[0])
    out = []
    for _, fingers in found:
        if fingers == "COMPLEMENT":
            out.append("COMPLEMENT")
        else:
            out.extend(f for f in fingers if f not in out)
    return out


def _state_in(clause):
    """The finger state a clause asserts, or None."""
    if re.search(HALF_CUES, clause):
        return "half"
    # curl before ext: "folded ... pointing down" style clauses are curls
    if re.search(r"\b(?:%s)\b" % CURL_CUES, clause):
        return "curl"
    if re.search(r"\b(?:%s)\b" % EXT_CUES, clause):
        return "ext"
    return None


def parse_hand_description(sentence):
    """One sentence -> (fingers dict, face_relation, n_explicit) or None.

    Grammar: clause-scoped binding. Each clause's finger mentions take the
    clause's state cue. "fist"/"open hand" set a whole-hand base first;
    "the other fingers" resolves to the complement of everything mentioned
    so far. Fingers never mentioned default to curl when the sentence names
    which fingers are extended (the way gesture prose is actually written),
    and the unmentioned thumb stays "any"."""
    low = sentence.lower()
    fingers = {}
    assigned = set()   # fingers explicitly named with a state
    explicit = 0
    pending_complement = None

    base = None
    if FIST_CUE.search(low):
        base = "curl"
    elif OPEN_CUE.search(low):
        base = "ext"
    if base:
        for f in NON_THUMB:
            fingers[f] = base

    # Two-level clause scan: a state cue carries forward across commas
    # within a major clause ("extending the thumb, index finger, and little
    # finger" - the later list items inherit "extending"), and resets at
    # contrast boundaries (";", "while", "whereas", "but").
    for major in _major_clauses(low):
        running = None
        for piece in re.split(r",|\bwith\b", major):
            state = _state_in(piece)
            if state is not None:
                running = state
            if running is None:
                continue
            for f in _fingers_in(piece):
                if f == "COMPLEMENT":
                    pending_complement = running
                    continue
                fingers[f] = running
                assigned.add(f)
                explicit += 1

    if pending_complement is not None:
        # "the other/remaining fingers" = non-thumb fingers not named so far
        for f in NON_THUMB:
            if f not in assigned:
                fingers[f] = pending_complement
                explicit += 1

    if not fingers or (explicit == 0 and base is None):
        return None

    # default: unmentioned fingers curl when the prose named the extended
    # ones; the thumb stays flexible unless stated
    if any(v == "ext" for k, v in fingers.items()):
        for f in NON_THUMB:
            fingers.setdefault(f, "curl")
    fingers.setdefault("thumb", "any")

    face_relation = None
    for pat, region, dist in FACE_REGIONS:
        m = re.search(r"(?:to|on|at|touch(?:ing|es)?|against|near|toward)"
                      r"[^.]{0,25}\b(?:%s)" % pat, low)
        if m:
            face_relation = {"region": region, "max_dist": dist}
            break

    orientation = None
    if re.search(r"(?:fingers?|tips?|hand)\s+point(?:ing|ed)?\s+up(?:ward)?"
                 r"|held\s+upright", low):
        orientation = "tips_up"

    palm = None
    if re.search(r"palm\s+(?:facing\s+)?(?:out(?:ward)?|forward|toward)"
                 r"|palm\s+(?:facing\s+)?the\s+(?:viewer|observer)", low):
        palm = "toward"
    elif re.search(r"back\s+of\s+the\s+hand\s+(?:facing|toward|to)"
                   r"|palm\s+(?:facing\s+)?(?:in(?:ward)?|the\s+signer)", low):
        palm = "away"

    return fingers, face_relation, orientation, explicit, base is not None, palm


# nationality/adjective words that are only a real name WITH their category
# noun kept ("French" -> useless, "French Salute" -> a real gesture)
_NEEDS_CATEGORY = {
    "french", "spanish", "german", "italian", "roman", "greek", "serbian",
    "russian", "british", "american", "chinese", "japanese", "indian",
    "arab", "arabic", "turkish", "scout", "military", "nazi", "royal",
    "three-finger", "two-finger", "polish", "hindu", "buddhist",
}


def extract_name(sentence, page_title=None):
    """Best gesture name for the sentence, or None."""
    candidates = []
    for m in re.finditer(
            r"\b((?:[A-Z][\w'-]+|ILY)(?:\s+[A-Za-z][\w'-]+){0,3}?)\s+"
            r"(sign|gesture|salute|greeting|mudra)\b", sentence):
        base, category = m.group(1), m.group(2)
        bwords = base.split()
        if bwords and bwords[0].lower() in {"the", "a", "an", "this", "it"}:
            bwords = bwords[1:]
        base = " ".join(bwords)
        # a single nationality/adjective word is only a name with its noun
        if len(bwords) == 1 and base.lower() in _NEEDS_CATEGORY:
            candidates.append(f"{base} {category.title()}")
        elif base:
            candidates.append(base)
    m = re.match(r"(?:The\s+)?([A-Z][\w'-]+(?:\s+[A-Za-z][\w'-]+){0,3}?)\s+"
                 r"(?:is|was|consists?|denotes?|means?)\b", sentence.strip())
    if m:
        candidates.append(m.group(1))
    for name in candidates:
        words = name.split()
        if words and words[0].lower() in {"the", "a", "an", "this", "it"}:
            words = words[1:]
        cleaned = " ".join(words).strip(" -'")
        if cleaned and _plausible_name(cleaned):
            return cleaned
    if (page_title and len(page_title.split()) <= 5
            and _plausible_name(page_title)):
        return page_title
    return None


def extract_hand_gestures(text, source_url, page_title=None):
    """All hand-gesture proposals a page's prose supports."""
    proposals = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) < 30 or len(sentence) > 600:
            continue
        if not GESTURE_CONTEXT.search(sentence.lower()):
            continue
        parsed = parse_hand_description(sentence)
        if parsed is None:
            continue
        fingers, face_relation, orientation, explicit, has_base, palm = parsed
        # confidence gate: enough explicit evidence to trust. A whole-hand
        # base (fist / open palm) counts - "a fist with the index extended"
        # is a complete description with one explicit finger.
        confidence = (explicit + (1 if has_base else 0)
                      + (2 if face_relation else 0))
        if confidence < 2:
            continue
        name = extract_name(sentence, page_title)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        spec = {"fingers": fingers}
        if face_relation:
            spec["face_relation"] = face_relation
        if orientation:
            spec["orientation"] = orientation
        if palm:
            spec["palm"] = palm
        proposals.append({
            "name": name.title() if name.islower() else name,
            "spec": spec,
            "notes": f"auto-extracted (confidence {confidence})",
            "evidence": " ".join(sentence.split())[:400],
            "source_url": source_url,
            "confidence": confidence,
            "kind": "hand",
        })
    return proposals


# ---------------------------------------------------------------------------
# Facial-expression grammar (FACS Action Units -> MediaPipe blendshapes)
# ---------------------------------------------------------------------------

# AU number -> blendshape key ("pair" keys mean average of Left/Right)
AU_BLENDSHAPES = {
    1: ["browInnerUp"], 2: ["browOuterUp"], 4: ["browDown"],
    5: ["eyeWide"], 6: ["cheekSquint"], 7: ["eyeSquint"],
    9: ["noseSneer"], 10: ["mouthUpperUp"], 12: ["mouthSmile"],
    14: ["mouthDimple"], 15: ["mouthFrown"], 17: ["mouthShrugLower"],
    18: ["mouthPucker"], 20: ["mouthStretch"], 23: ["mouthPress"],
    24: ["mouthPress"], 25: ["jawOpen"], 26: ["jawOpen"],
    28: ["mouthRollLower"], 43: ["eyeBlink"], 45: ["eyeBlink"],
}

MUSCLE_PHRASES = [
    (r"inner\s+brow\s+rais|eyebrows?\s+(?:pulled|drawn)\s+(?:up|together)",
     ["browInnerUp"]),
    (r"(?:raised|raising|lifted)\s+(?:eye)?brows?|brow\s+rais",
     ["browInnerUp", "browOuterUp"]),
    (r"(?:lowered|furrowed|knitted|drawn[- ]down)\s+(?:eye)?brows?"
     r"|brow\s+lower|frown(?:ing)?\s+brow", ["browDown"]),
    (r"widened?\s+eyes?|eyes?\s+wide|eye\s+widen|upper\s+lid\s+rais",
     ["eyeWide"]),
    (r"squint|lid\s+tighten|narrowed\s+eyes", ["eyeSquint"]),
    (r"cheek\s+rais", ["cheekSquint"]),
    (r"nose\s+wrinkl|wrinkled\s+nose", ["noseSneer"]),
    (r"lip\s+corner\s+pull|smil(?:e|ing)|grin", ["mouthSmile"]),
    (r"lip\s+corner\s+depress|down[- ]?turned\s+(?:mouth|lips)"
     r"|corners?\s+of\s+the\s+mouth\s+(?:turned\s+)?down", ["mouthFrown"]),
    (r"(?:pursed|puckered)\s+lips|lip\s+pucker", ["mouthPucker"]),
    (r"pressed\s+lips|lip\s+press|lips\s+pressed", ["mouthPress"]),
    (r"(?:dropped|open(?:ed)?)\s+(?:jaw|mouth)|jaw\s+drop|mouth\s+open",
     ["jawOpen"]),
    (r"lip\s+stretch|stretched\s+(?:lips|mouth)", ["mouthStretch"]),
    (r"upper\s+lip\s+rais", ["mouthUpperUp"]),
    (r"tongue\s+(?:out|show|protru)", ["tongueOut"]),
]

EXPRESSION_NAME_WORDS = (
    r"disgust|sadness|sad|fear|fearful|joy|surprise[d]?|anger|angry"
    r"|contempt|confusion|confused|boredom|bored|pain|interest|awe"
    r"|embarrassment|shame|pride|amusement|skepticism|doubt|disbelief"
    r"|determination|grimace|pout|sneer|smirk|scowl|wince")

EXPRESSION_CONTEXT = re.compile(
    r"expression|facial|face\b|emotion|\bAU\s?\d|action\s+unit")


def extract_expressions(text, source_url):
    """Facial-expression proposals: name + blendshape thresholds."""
    proposals = []
    seen = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) < 25 or len(sentence) > 600:
            continue
        low = sentence.lower()
        if not EXPRESSION_CONTEXT.search(low):
            continue

        blends = {}
        for m in re.finditer(r"\bAU\s?(\d{1,2})\b", sentence):
            for key in AU_BLENDSHAPES.get(int(m.group(1)), []):
                blends[key] = 0.4
        for pat, keys in MUSCLE_PHRASES:
            if re.search(pat, low):
                for key in keys:
                    blends.setdefault(key, 0.4)
        if len(blends) < 2:
            continue  # one muscle is not an expression signature

        m = re.search(r"\b(%s)\b" % EXPRESSION_NAME_WORDS, low)
        if not m:
            continue
        name = m.group(1).title()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        proposals.append({
            "name": name,
            "face_spec": blends,
            "notes": f"auto-extracted expression ({len(blends)} muscle cues)",
            "evidence": " ".join(sentence.split())[:400],
            "source_url": source_url,
            "confidence": len(blends),
            "kind": "expression",
        })
    return proposals


def extract_all(text, source_url, page_title=None):
    """Every proposal - hand gestures and expressions - a page supports."""
    return (extract_hand_gestures(text, source_url, page_title)
            + extract_expressions(text, source_url))
