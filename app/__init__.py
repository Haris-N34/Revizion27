"""Octaby app. Runs before any app module, so env set here applies before mediapipe loads."""

import os

# Silence MediaPipe/TFLite info + warning spam (the I0000/W0000 lines).
for _key in ("GLOG_minloglevel", "ABSL_MIN_LOG_LEVEL", "TF_CPP_MIN_LOG_LEVEL"):
    os.environ.setdefault(_key, "2")
