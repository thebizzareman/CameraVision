"""
Runtime conflict-resolution filter implementing RULEBOOK.md.

Every frame's raw gesture/expression observations pass through
ConflictFilter.apply() BEFORE temporal voting. Rule IDs match RULEBOOK.md
one-to-one; rules R3/R4/R9 live in the classifiers themselves and are only
documented there. Each suppression produces a human-readable note that the
app shows in the on-screen event feed, so the filter's decisions are
auditable live.
"""

# Frames within which respect/aggression gestures are considered contradictory
CONTRADICTION_WINDOW = 45  # ~1.5 s at 30 fps


class ConflictFilter:
    def __init__(self, window_frames=CONTRADICTION_WINDOW):
        self._window = window_frames
        self._frame = 0
        self._last_seen = {}   # name -> frame index when last kept
        self._prev_notes = set()

    def apply(self, observations, hands, face=None):
        """observations: list of (name, hand_index_or_None).
        Returns (kept_names, new_notes) - notes only on their first frame so
        the event feed isn't flooded while a conflict persists."""
        self._frame += 1
        names = {n for n, _ in observations}
        by_hand = {}
        for n, h in observations:
            if h is not None:
                by_hand.setdefault(n, set()).add(h)

        drop = set()
        notes = []

        # R1: Kapak uses both hands - no face+hand gesture can coexist
        if "KAPAK" in names:
            for g in ("EYVALLAH", "YEMEZLER"):
                if g in names:
                    drop.add(g)
                    notes.append(f"R1: {g} dropped - KAPAK occupies both hands")

        # R2: the same hand cannot touch the eye and the chest at once
        if ("EYVALLAH" in names and "YEMEZLER" in names
                and by_hand.get("EYVALLAH", set()) & by_hand.get("YEMEZLER", set())):
            drop.add("EYVALLAH")
            notes.append("R2: EYVALLAH dropped - same hand is at the eye")

        # R5: during the money-rub motion, a static OK is a misread frame
        if "HESAP LUTFEN" in names and "OK" in names:
            drop.add("OK")
            notes.append("R5: OK dropped - money-rub motion in progress")

        # R6: respect (Eyvallah) and aggression (Middle Finger) in one
        # temporal window means keypoint hallucination - trust neither
        both_now = {"EYVALLAH", "Middle Finger"} <= (names - drop)
        recent = lambda n: self._frame - self._last_seen.get(n, -10**9) <= self._window
        if both_now:
            drop.update(("EYVALLAH", "Middle Finger"))
            notes.append("R6: EYVALLAH + Middle Finger together - keypoint error, both dropped")
        elif "Middle Finger" in names - drop and recent("EYVALLAH"):
            drop.add("Middle Finger")
            notes.append("R6: Middle Finger dropped - contradicts recent EYVALLAH")
        elif "EYVALLAH" in names - drop and recent("Middle Finger"):
            drop.add("EYVALLAH")
            notes.append("R6: EYVALLAH dropped - contradicts recent Middle Finger")

        # R7: pucker and open jaw are antagonistic - keep the stronger signal
        if "Kiss" in names and "Mouth Open" in names:
            blend = getattr(face, "blendshapes", {}) or {}
            loser = ("Kiss" if blend.get("jawOpen", 0) >= blend.get("mouthPucker", 0)
                     else "Mouth Open")
            drop.add(loser)
            notes.append(f"R7: {loser} dropped - antagonistic mouth shapes")

        # R8: a protruding tongue reshapes the lips into a false pucker
        if "Tongue Out" in names and "Kiss" in names and "Kiss" not in drop:
            drop.add("Kiss")
            notes.append("R8: Kiss dropped - tongue is out")

        kept = names - drop
        for n in kept:
            self._last_seen[n] = self._frame

        new_notes = [n for n in notes if n not in self._prev_notes]
        self._prev_notes = set(notes)
        return kept, new_notes
