"""
Portable-learning CLI: carry what CameraVision has learned between machines.

  python manage.py export mylearning.zip     # bundle config + corrections + DB
  python manage.py import mylearning.zip      # restore it on another machine

The bundle contains config.json (your calibration + letter biases), the
correction log, and gesture_db.json (the grown database including everything
the autonomous researcher has added). Import only extracts those known
files - it never writes arbitrary paths from the archive.
"""

import sys

import config


def main(argv):
    if len(argv) != 2 or argv[0] not in ("export", "import"):
        print(__doc__)
        return 1
    action, path = argv
    if action == "export":
        files = config.export_learning(path)
        print(f"Exported {len(files)} file(s) to {path}: {', '.join(files)}")
    else:
        files = config.import_learning(path)
        print(f"Restored {len(files)} file(s) from {path}: {', '.join(files)}")
        print("Restart the app (or press R) to pick up the restored data.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
