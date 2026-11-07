"""
Offline accuracy evaluation over recorded hands.

Turns tuning from anecdote into measurement:

  1. Record labeled clips:   python app.py --record A        (sign the letter A)
                             python app.py --record thumbs-up
     Each session writes recordings/<label>_<timestamp>.jsonl - one JSON
     line per frame with the 21 hand landmarks.

  2. Score everything:       python eval.py
     Rebuilds each frame's Hand, runs the ASL classifier (single-letter
     labels) or the gesture classifier (word labels), and prints a
     per-label accuracy table with the most common confusion.
"""

import glob
import json
import os
import sys
from collections import Counter, namedtuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(BASE_DIR, "recordings")
LM = namedtuple("LM", "x y z")
SHAPE = (480, 640, 3)


def main():
    sys.path.insert(0, BASE_DIR)
    from gestures import Hand
    import signs

    files = sorted(glob.glob(os.path.join(REC_DIR, "*.jsonl")))
    if not files:
        print(f"no recordings in {REC_DIR}; record with: "
              f"python app.py --record <label>")
        return

    rows = []
    for path in files:
        label = os.path.basename(path).rsplit("_", 1)[0]
        got = Counter()
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                lms = [LM(*p) for p in rec["lms"]]
                hand = Hand(lms, rec.get("handedness", "Right"), SHAPE)
                if len(label) == 1:            # ASL letter
                    got[signs.classify(hand) or "-"] += 1
                else:                          # named gesture
                    got[hand.gesture or "-"] += 1
                n += 1
        if not n:
            continue
        want = label.upper() if len(label) == 1 else label
        correct = sum(v for k, v in got.items()
                      if k.lower() == want.lower())
        top_wrong = next((f"{k} x{v}" for k, v in got.most_common()
                          if k.lower() != want.lower()), "-")
        rows.append((want, n, correct / n, top_wrong))

    print(f"{'label':<16}{'frames':>7}{'accuracy':>10}   top confusion")
    print("-" * 55)
    for label, n, acc, wrong in sorted(rows):
        print(f"{label:<16}{n:>7}{acc:>9.0%}   {wrong}")
    total = sum(n for _, n, _, _ in rows)
    weighted = sum(n * a for _, n, a, _ in rows) / max(total, 1)
    print("-" * 55)
    print(f"{'OVERALL':<16}{total:>7}{weighted:>9.0%}")


if __name__ == "__main__":
    main()
