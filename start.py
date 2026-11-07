#!/usr/bin/env python3
import sys
import os

# Ensure the CameraVision directory is in the path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Run the app
if __name__ == "__main__":
    import app
    app.main()
