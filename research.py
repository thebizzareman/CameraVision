"""
Online gesture research pipeline.

Fetches gesture reference pages (via the gstack browser when connected,
urllib otherwise), extracts candidate gestures using a curated rule library
that maps prose descriptions to declarative detection specs, then runs every
candidate through a validation gate before anything touches the live
database:

  1. expressibility  - can the description be stated in the spec schema?
  2. duplicate check - finger-signature collision against every known entry
                       (a duplicate with a different name becomes a synonym
                       on the existing entry, not a new gesture)
  3. synthetic test  - a synthetic hand built from the spec must actually
                       trigger match_spec(); if the builtin classifier also
                       claims that hand, an override is recorded
  4. provenance      - merged entries carry source URL + retrieval date

Validated entries merge into gesture_db.json (status "active"); everything
else lands in research_cache/proposals_pending.json with the reason, for
human review. The live app never depends on the network - this pipeline
only grows the local database.

Usage:
  python research.py                     # fetch + validate + merge defaults
  python research.py --url <URL>        # research a specific page
  python research.py --offline <file> --source-url <URL>
  python research.py --dry-run          # validate but don't merge
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import namedtuple
from datetime import date
from types import SimpleNamespace

import numpy as np

import gesture_db
import prose2spec
from gestures import Hand

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "research_cache")
PAGE_CACHE = os.path.join(CACHE_DIR, "pages")
PENDING_PATH = os.path.join(CACHE_DIR, "proposals_pending.json")
CACHE_TTL = 7 * 86400          # a week: re-visits are free, content stays fresh
POLITENESS_DELAY = 1.0         # seconds between live requests

DEFAULT_SOURCES = [
    "https://en.wikipedia.org/wiki/List_of_gestures",
]

# Autonomous mode (--auto) searches these topics by itself - English and
# Turkish Wikipedia for gesture/sign/expression articles, arXiv for papers.
SEARCH_TOPICS = [
    ("en", "list of gestures"),
    ("en", "hand gesture"),
    ("en", "manual alphabet"),
    ("en", "sign language"),
    ("en", "facial expression"),
    ("en", "facial action coding system"),
    ("en", "mudra"),
    ("en", "salute"),
    ("tr", "el hareketleri"),
    ("tr", "isaret dili"),
]
ARXIV_QUERIES = [
    "hand gesture recognition finger",
    "facial expression recognition action units",
]

BROWSE_CANDIDATES = [
    os.path.expanduser(r"~\.claude\skills\gstack\browse\dist\browse.exe"),
    os.path.expanduser(r"~\.claude\skills\gstack\browse\dist\browse"),
]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch(url):
    """Page text via the gstack browser if available, else urllib."""
    for browse in BROWSE_CANDIDATES:
        if not os.path.isfile(browse):
            continue
        try:
            subprocess.run([browse, "goto", url], capture_output=True,
                           timeout=60, check=True)
            out = subprocess.run([browse, "text"], capture_output=True,
                                 timeout=60, check=True)
            text = out.stdout.decode("utf-8", errors="replace")
            if len(text) > 500:
                print(f"  fetched via gstack browser: {len(text)} chars")
                return text
        except (subprocess.SubprocessError, OSError):
            continue
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 "
                                 "(CameraVision gesture research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    print(f"  fetched via urllib: {len(text)} chars")
    return text


# ---------------------------------------------------------------------------
# Autonomous source discovery (Wikipedia search API + arXiv), with an
# on-disk page cache and politeness delays. The pipeline finds its own
# sources instead of depending on a manually curated URL list.
# ---------------------------------------------------------------------------

def _http(url, _retry=True):
    """Raw GET with cache, politeness delay, and one 429 backoff retry."""
    os.makedirs(PAGE_CACHE, exist_ok=True)
    key = hashlib.md5(url.encode("utf-8")).hexdigest()
    path = os.path.join(PAGE_CACHE, key + ".txt")
    if (os.path.isfile(path)
            and time.time() - os.path.getmtime(path) < CACHE_TTL):
        with open(path, encoding="utf-8") as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": "CameraVision/1.0 "
                                 "(gesture research; local hobby project)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and _retry:
            print("    (rate limited - backing off 15s)")
            time.sleep(15)
            return _http(url, _retry=False)
        raise
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    time.sleep(POLITENESS_DELAY)
    return body


def wiki_search(term, lang="en", limit=4):
    """Wikipedia article titles matching a search term."""
    url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
           f"&srsearch={urllib.parse.quote(term)}&srlimit={limit}"
           f"&format=json")
    data = json.loads(_http(url))
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def wiki_extract(title, lang="en"):
    """Plain-text article body + canonical URL."""
    url = (f"https://{lang}.wikipedia.org/w/api.php?action=query"
           f"&prop=extracts&explaintext=1&redirects=1"
           f"&titles={urllib.parse.quote(title)}&format=json")
    data = json.loads(_http(url))
    pages = data.get("query", {}).get("pages", {})
    text = " ".join(p.get("extract", "") for p in pages.values())
    page_url = (f"https://{lang}.wikipedia.org/wiki/"
                f"{urllib.parse.quote(title.replace(' ', '_'))}")
    return text, page_url


def arxiv_abstracts(query, limit=6):
    """(title, abstract, link) tuples from the arXiv API."""
    url = (f"http://export.arxiv.org/api/query?search_query="
           f"{urllib.parse.quote('all:' + query)}&max_results={limit}")
    body = _http(url)
    out = []
    for m in re.finditer(r"<entry>(.*?)</entry>", body, re.S):
        entry = m.group(1)
        title = re.search(r"<title>(.*?)</title>", entry, re.S)
        summary = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        link = re.search(r'<id>(.*?)</id>', entry, re.S)
        if title and summary:
            out.append((" ".join(title.group(1).split()),
                        " ".join(summary.group(1).split()),
                        link.group(1).strip() if link else "arxiv.org"))
    return out


def discover(max_pages=14):
    """Autonomously assemble (text, url, title) documents to research."""
    docs, seen = [], set()
    for lang, term in SEARCH_TOPICS:
        if len(docs) >= max_pages:
            break
        try:
            titles = wiki_search(term, lang)
        except Exception as exc:
            print(f"  search '{term}' ({lang}) failed: {exc}")
            continue
        for title in titles[:2]:
            if (lang, title) in seen or len(docs) >= max_pages:
                continue
            seen.add((lang, title))
            try:
                text, url = wiki_extract(title, lang)
            except Exception as exc:
                print(f"  extract '{title}' failed: {exc}")
                continue
            if len(text) > 400:
                print(f"  [{len(docs) + 1}] wikipedia({lang}): {title} "
                      f"({len(text)} chars)")
                docs.append((text, url, title))
    for query in ARXIV_QUERIES:
        if len(docs) >= max_pages + len(ARXIV_QUERIES):
            break
        try:
            for title, abstract, link in arxiv_abstracts(query):
                print(f"  [{len(docs) + 1}] arxiv: {title[:60]}")
                docs.append((abstract, link, None))
        except Exception as exc:
            print(f"  arxiv '{query}' failed: {exc}")
    return docs


# ---------------------------------------------------------------------------
# Extraction rule library: prose pattern -> detection spec.
# A rule only fires when its evidence pattern appears in the page text, so
# the database grows from documented descriptions, not invention.
# ---------------------------------------------------------------------------

Rule = namedtuple("Rule", "name pattern spec notes")

RULE_LIBRARY = [
    Rule("Shaka",
         r"shaka sign[^.]*thumb and (?:little finger|pinky)",
         {"fingers": {"thumb": "ext", "index": "curl", "middle": "curl",
                      "ring": "curl", "pinky": "ext"}},
         "Hawaii/surf friendship sign"),
    Rule("Finger Gun",
         r"finger gun",
         {"fingers": {"thumb": "ext", "index": "ext", "middle": "curl",
                      "ring": "curl", "pinky": "curl"},
          "overrides": ["Pointing"]},
         "thumb up + index out; builtin Pointing ignores the thumb"),
    Rule("Three-Finger Salute",
         r"three-finger salute[^.]*thumb, index,? and middle",
         {"fingers": {"thumb": "ext", "index": "ext", "middle": "ext",
                      "ring": "curl", "pinky": "curl"},
          "overrides": ["Peace"]},
         "Serbian salute; builtin Peace ignores the thumb"),
    Rule("Grey Wolf Salute",
         r"grey wolf salute[^.]*little finger and index",
         {"fingers": {"thumb": "any", "index": "ext", "middle": "curl",
                      "ring": "curl", "pinky": "ext"}},
         "expected duplicate of Rock -> synonym"),
    Rule("Two-Finger Salute",
         r"two-finger salute[^.]*middle and index",
         {"fingers": {"thumb": "any", "index": "ext", "middle": "ext",
                      "ring": "curl", "pinky": "curl"}},
         "expected duplicate of Peace -> synonym"),
    Rule("Salute",
         r"salute[^.]*(?:brow|forehead)",
         {"fingers": {"thumb": "any", "index": "ext", "middle": "ext",
                      "ring": "ext", "pinky": "any"},
          "face_relation": {"region": "forehead", "max_dist": 1.3}},
         "flat hand raised to the brow"),
    Rule("Vulcan Salute",
         r"vulcan salute",
         {"fingers": {"thumb": "any", "index": "ext", "middle": "ext",
                      "ring": "ext", "pinky": "ext"},
          "spread": {"pair": ["middle", "ring"], "min_ratio": 1.7},
          "overrides": ["Open Palm"]},
         "expressible since the spread primitive: V split between "
         "middle and ring fingers"),
]


def extract(text, source_url):
    low = text.lower()
    proposals = []
    for rule in RULE_LIBRARY:
        m = re.search(rule.pattern, low)
        if not m:
            continue
        start = max(0, m.start() - 40)
        evidence = " ".join(text[start:m.end() + 160].split())
        proposals.append({
            "name": rule.name,
            "spec": rule.spec,
            "notes": rule.notes,
            "evidence": evidence,
            "source_url": source_url,
        })
    return proposals


# ---------------------------------------------------------------------------
# Synthetic hand construction for the validation gate
# ---------------------------------------------------------------------------

LM = namedtuple("LM", "x y")
_SHAPE = (480, 640, 3)


def synthetic_hand(fingers, spec=None):
    """Builds a gestures.Hand posed to the given finger states. When the
    spec uses geometric primitives (spread, tip_contact), the pose honors
    them so the validation gate can exercise the full detector."""
    xs = {"index": 0.47, "middle": 0.50, "ring": 0.53, "pinky": 0.56}
    drift = {k: 0.0 for k in xs}
    if spec and spec.get("spread"):
        fa, fb = spec["spread"]["pair"]
        if fa in drift and fb in drift:
            drift[fa] -= 0.035
            drift[fb] += 0.035
    lms = [LM(0.50, 0.62)]  # wrist
    if fingers.get("thumb") == "ext":
        lms += [LM(0.45, 0.59), LM(0.42, 0.55), LM(0.40, 0.52), LM(0.38, 0.49)]
    else:
        lms += [LM(0.455, 0.60), LM(0.45, 0.58), LM(0.46, 0.565), LM(0.47, 0.56)]
    for key in ("index", "middle", "ring", "pinky"):
        x = xs[key]
        tx = x + drift[key]
        if fingers.get(key) == "ext":
            lms += [LM(x, 0.52), LM(x, 0.46), LM((x + tx) / 2, 0.41),
                    LM(tx, 0.37)]
        else:
            lms += [LM(x, 0.52), LM(x, 0.49), LM(x, 0.505), LM(x, 0.54)]
    return Hand(lms, "Right", _SHAPE)


def _fake_face():
    return SimpleNamespace(center=np.array([320.0, 150.0]),
                           chin=np.array([320.0, 200.0]),
                           forehead=np.array([320.0, 100.0]),
                           width=120.0, height=140.0)


def _shift_hand_to(hand_fingers, point):
    """Synthetic hand translated so its center sits on `point` (px)."""
    base = synthetic_hand(hand_fingers)
    dx = (point[0] - base.center[0]) / 640.0
    dy = (point[1] - base.center[1]) / 480.0
    lms = [LM(p[0] / 640.0 + dx, p[1] / 480.0 + dy) for p in base.points]
    return Hand(lms, "Right", _SHAPE)


def _region_point(face, region):
    if region == "forehead":
        return face.forehead
    if region == "chest":
        return face.chin + np.array([0.0, face.height * 1.2])
    return face.center


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def validate(proposal, entries):
    """Returns (verdict, detail): verdict in
    merge | synonym | pending | reject."""
    name = proposal["name"]
    spec = proposal["spec"]

    # 1. expressibility
    if spec is None:
        return "pending", proposal["notes"]

    # existing name? Compare canonically: "OK gesture" == "OK", "Corna
    # sign" == "Corna" - the suffix words carry no identity.
    def _canon(n):
        words = [w for w in n.lower().split()
                 if w not in ("gesture", "sign", "salute", "greeting",
                              "the", "hand")]
        return " ".join(words) or n.lower()

    for e in entries:
        known = [e["name"]] + e.get("synonyms", [])
        if any(_canon(name) == _canon(k) for k in known):
            return "reject", f"already known as {e['name']}"

    # 2. duplicate signature -> synonym on the existing entry, UNLESS the
    # proposal is strictly more specific (concrete state where the existing
    # signature says "any"): then it's a finer, distinct gesture - e.g. ILY
    # (thumb+index+pinky ext) vs Rock (thumb any) - and merges with an
    # override instead of vanishing into a synonym.
    def _stricter(a, b):
        return (any(y == "any" and x != "any" for x, y in zip(a, b))
                and not any(x == "any" and y != "any" for x, y in zip(a, b)))

    sig = tuple(spec["fingers"].get(k, "any") for k in gesture_db.FINGER_KEYS)
    more_specific = bool(spec.get("face_relation") or spec.get("orientation"))
    for e in entries:
        if e["name"] in spec.get("overrides", []):
            continue  # declared override: proposal is the more specific shape
        esig = gesture_db.signature_of(e)
        if esig and gesture_db.signatures_collide(sig, esig):
            e_specific = bool(e.get("detector", {}).get("face_relation"))
            if more_specific and not e_specific:
                continue  # e.g. Salute vs Open Palm: face relation disambiguates
            if _stricter(sig, esig):
                overrides = spec.setdefault("overrides", [])
                if e["name"] not in overrides:
                    overrides.append(e["name"])
                continue
            return "synonym", e["name"]

    # 3. synthetic detection test
    entry = proposal_to_entry(proposal)
    hand = synthetic_hand(spec["fingers"], spec)
    face = _fake_face() if spec.get("face_relation") else None
    if face is not None:
        region = spec["face_relation"].get("region", "face")
        hand = _shift_hand_to(spec["fingers"], _region_point(face, region))
    if not gesture_db.match_spec(entry, hand, face):
        return "reject", "synthetic hand failed to trigger the spec"
    builtin = hand.gesture
    overrides = spec.get("overrides", [])
    if builtin and builtin not in overrides and not spec.get("face_relation"):
        overrides.append(builtin)
        spec["overrides"] = overrides
        proposal["auto_override"] = builtin

    return "merge", entry


def proposal_to_entry(p):
    det = {"kind": "spec", "fingers": p["spec"]["fingers"]}
    for k in ("orientation", "face_relation", "overrides", "palm",
              "spread", "crossed", "thumb_between", "tip_contact"):
        if p["spec"].get(k):
            det[k] = p["spec"][k]
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", p["name"].lower()).strip("-"),
        "name": p["name"],
        "type": "face_hand" if p["spec"].get("face_relation") else "single_hand",
        "origin": "research",
        "description": p["evidence"][:300],
        "display": "card",
        "status": "active",
        "detector": det,
        "sources": [{"url": p["source_url"], "retrieved": str(date.today())}],
        "notes": p["notes"],
    }


# ---------------------------------------------------------------------------
# Expression validation (face_spec proposals from prose2spec)
# ---------------------------------------------------------------------------

_BLEND_PAIRS = {"browDown", "browOuterUp", "cheekSquint", "eyeBlink",
                "eyeLookDown", "eyeLookIn", "eyeLookOut", "eyeLookUp",
                "eyeSquint", "eyeWide", "mouthDimple", "mouthFrown",
                "mouthLowerDown", "mouthPress", "mouthSmile", "mouthStretch",
                "mouthUpperUp", "noseSneer"}
_BLEND_SINGLES = {"browInnerUp", "cheekPuff", "jawForward", "jawLeft",
                  "jawOpen", "jawRight", "mouthClose", "mouthFunnel",
                  "mouthLeft", "mouthPucker", "mouthRight", "mouthRollLower",
                  "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
                  "tongueOut"}
VALID_BLENDS = _BLEND_PAIRS | _BLEND_SINGLES


def expression_to_entry(p):
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", p["name"].lower()).strip("-"),
        "name": p["name"],
        "type": "expression",
        "origin": "research",
        "description": p["evidence"][:300],
        "display": "face_label",
        "status": "active",
        "detector": {"kind": "face_spec", "blends": p["face_spec"]},
        "sources": [{"url": p["source_url"], "retrieved": str(date.today())}],
        "notes": p["notes"],
    }


def validate_expression(prop, entries):
    """(verdict, detail) for a facial-expression proposal."""
    import faces
    blends = prop["face_spec"]
    bad = [k for k in blends if k not in VALID_BLENDS]
    if bad:
        return "reject", f"unknown blendshape key(s): {', '.join(bad)}"

    builtin = {n.lower() for n in faces.EXPRESSION_RULES}
    builtin |= {n.lower() for n in faces.EMOTION_NAMES}
    if prop["name"].lower() in builtin:
        return "reject", "already a builtin expression/emotion"
    for e in entries:
        if e["name"].lower() == prop["name"].lower():
            return "reject", f"already known as {e['name']}"
        det = e.get("detector", {})
        if (det.get("kind") == "face_spec"
                and set(det.get("blends", {})) == set(blends)):
            return "synonym", e["name"]

    # synthetic activation test: fires on the described muscles, not on a
    # neutral face
    entry = expression_to_entry(prop)
    if not gesture_db.match_face_spec(entry, {k: 0.9 for k in blends}):
        return "reject", "synthetic activation failed to trigger the spec"
    if gesture_db.match_face_spec(entry, {}):
        return "reject", "spec fires on a neutral face"
    return "merge", entry


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(sources=None, offline=None, source_url=None, dry_run=False,
        auto=False, max_pages=14):
    os.makedirs(CACHE_DIR, exist_ok=True)
    entries = gesture_db.load()
    texts = []   # (text, url, page_title)
    if offline:
        with open(offline, encoding="utf-8") as f:
            texts.append((f.read(), source_url or f"file://{offline}", None))
    elif auto:
        print("Discovering sources (Wikipedia en/tr + arXiv)...")
        texts = discover(max_pages)
    else:
        for url in (sources or DEFAULT_SOURCES):
            print(f"Fetching {url}")
            try:
                texts.append((fetch(url), url, None))
            except Exception as exc:
                print(f"  FAILED: {exc}")

    # Gather proposals: curated rule library + the generative grammar.
    # Dedupe by name within the run, keeping the higher-confidence one.
    proposals = {}
    for text, url, title in texts:
        for prop in extract(text, url):
            prop.setdefault("kind", "hand")
            proposals.setdefault(prop["name"].lower(), prop)
        for prop in prose2spec.extract_all(text, url, title):
            key = prop["name"].lower()
            old = proposals.get(key)
            # curated rule-library proposals (no confidence key) win over
            # grammar extractions; among grammar ones, higher confidence wins
            if old is None or ("confidence" in old
                               and prop["confidence"] > old["confidence"]):
                proposals[key] = prop

    merged, synonyms, pending, rejected = [], [], [], []
    for prop in proposals.values():
        url = prop["source_url"]
        if prop.get("kind") == "expression":
            verdict, detail = validate_expression(prop, entries)
        else:
            verdict, detail = validate(prop, entries)
        if verdict == "merge":
            entries.append(detail)
            note = f" (auto-override: {prop['auto_override']})" \
                if prop.get("auto_override") else ""
            merged.append(f"{prop['name']}{note}")
        elif verdict == "synonym":
            for e in entries:
                if e["name"] == detail:
                    syns = e.setdefault("synonyms", [])
                    if prop["name"] not in syns:
                        syns.append(prop["name"])
                        e.setdefault("sources", []).append(
                            {"url": url, "retrieved": str(date.today()),
                             "note": f"synonym: {prop['name']}"})
            synonyms.append(f"{prop['name']} -> {detail}")
        elif verdict == "pending":
            pending.append({**prop, "reason": detail})
        else:
            rejected.append(f"{prop['name']}: {detail}")

    print("\n=== RESEARCH REPORT ===")
    print(f"merged ({len(merged)}):   " + (", ".join(merged) or "-"))
    print(f"synonyms ({len(synonyms)}): " + (", ".join(synonyms) or "-"))
    print(f"pending ({len(pending)}):  "
          + (", ".join(p["name"] for p in pending) or "-"))
    print(f"rejected ({len(rejected)}): " + ("; ".join(rejected) or "-"))

    summary = {"merged": len(merged), "synonyms": len(synonyms),
               "pending": len(pending), "rejected": len(rejected)}
    if dry_run:
        print("\ndry run - database not modified")
        return summary

    if merged or synonyms:
        gesture_db.save(entries)
        print(f"\ngesture_db.json updated ({len(entries)} entries)")
    if pending:
        existing = []
        if os.path.isfile(PENDING_PATH):
            with open(PENDING_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        known = {p["name"] for p in existing}
        existing += [p for p in pending if p["name"] not in known]
        # drop staged proposals that have since become real entries
        # (e.g. Vulcan Salute once the spread primitive made it expressible)
        db_names = {e["name"].lower() for e in entries}
        existing = [p for p in existing if p["name"].lower() not in db_names]
        with open(PENDING_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"{len(pending)} proposal(s) staged in {PENDING_PATH}")
    return summary


def list_pending():
    """Staged proposals (for the web review UI). [] if none."""
    if not os.path.isfile(PENDING_PATH):
        return []
    try:
        with open(PENDING_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def resolve_pending(name, action):
    """Approve or reject one staged proposal by name (non-interactive, for
    the dashboard). Returns {ok, msg}."""
    pending = list_pending()
    if not any(p["name"] == name for p in pending):
        return {"ok": False, "msg": f"no pending proposal '{name}'"}
    entries = gesture_db.load()
    keep, result = [], {"ok": False, "msg": "unchanged"}
    for p in pending:
        if p["name"] != name:
            keep.append(p)
            continue
        if action == "reject":
            result = {"ok": True, "msg": f"{name} rejected"}
            continue
        verdict, detail = (validate_expression(p, entries)
                           if p.get("kind") == "expression"
                           else validate(p, entries))
        if verdict == "merge":
            entries.append(detail)
            gesture_db.save(entries)
            result = {"ok": True, "msg": f"{name} merged into the database"}
        else:
            keep.append(p)
            result = {"ok": False, "msg": f"{name}: {verdict} ({detail})"}
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)
    return result


def review_pending():
    """Interactive approve/reject for staged proposals."""
    if not os.path.isfile(PENDING_PATH):
        print("nothing pending")
        return
    with open(PENDING_PATH, encoding="utf-8") as f:
        pending = json.load(f)
    entries = gesture_db.load()
    keep, changed = [], False
    for i, p in enumerate(pending):
        print(f"\n--- {p['name']}")
        print(f"    reason:   {p.get('reason', '?')}")
        print(f"    evidence: {p.get('evidence', '')[:180]}")
        print(f"    spec:     {json.dumps(p.get('spec') or p.get('face_spec'))}")
        ans = input("    [a]pprove / [r]eject / [s]kip / [q]uit? ").strip().lower()
        if ans == "q":
            keep.extend(pending[i:])
            break
        if ans == "a":
            if p.get("kind") == "expression":
                verdict, detail = validate_expression(p, entries)
            else:
                verdict, detail = validate(p, entries)
            if verdict == "merge":
                entries.append(detail)
                changed = True
                print(f"    merged: {p['name']}")
            else:
                print(f"    still not mergeable ({verdict}: {detail}) - kept")
                keep.append(p)
        elif ans == "r":
            changed = True
            print("    dropped")
        else:
            keep.append(p)
    if changed:
        gesture_db.save(entries)
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)
    print(f"\n{len(keep)} proposal(s) remain pending")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--url", action="append", help="research a specific page")
    ap.add_argument("--offline", help="use a cached text file instead of fetching")
    ap.add_argument("--source-url", help="provenance URL for --offline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="autonomous mode: discover sources by itself "
                         "(Wikipedia en/tr search + arXiv)")
    ap.add_argument("--max-pages", type=int, default=14,
                    help="page budget for --auto (default 14)")
    ap.add_argument("--review", action="store_true",
                    help="interactively approve/reject pending proposals")
    args = ap.parse_args()
    if args.review:
        review_pending()
    else:
        run(sources=args.url, offline=args.offline,
            source_url=args.source_url, dry_run=args.dry_run,
            auto=args.auto, max_pages=args.max_pages)
