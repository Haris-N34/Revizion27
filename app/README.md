# REVISION app (Harim)
1. Install app dependencies: `python -m pip install -r app/requirements.txt`.
2. Web UI: `python -m app.web`, then open **http://127.0.0.1:8765** on the laptop. Hosted page: **https://revision-live-guidance.vercel.app** (laptop and temporary tunnel must stay running).
3. Hosted **This device** mode uses that visitor's webcam: enable permission, prefer DroidCam when available, otherwise use the browser default; the camera picker remembers each browser's choice. **Laptop + wristband** uses `camera_index` in local `config.yaml`; Arduino uses `serial_port` and `baud`. Rotate a sideways feed with **Rotate camera** or launch the laptop with `--rotation 90`.
4. Choose a target or press the **wristband button (pin 12)**, speak, then pause. **Stop guidance** / **Esc** stops motors; closing or hiding the page pauses guidance. Speech uses the laptop microphone/speakers; no spoken directions.
5. Standalone demo: `python -m app.web --demo` (synthetic camera and mock hardware, no voice); `--mock-band`, `--no-voice`, `--video FILE`, and `--port 8766` are also available.
6. Desktop: `python -m app.main`; optional `--target can`, `--mock-band`, `--no-voice`, `--video FILE`, `--no-window`, `--say-directions on`. Quit with **q**; stop the web server with **Ctrl+C**.
7. Tests without hardware: `python -m pytest app`. Calibrate with `python -m app.calibrate`. Whisper `base.en` downloads once; speech uses Windows SAPI, macOS `say`, or Linux pyttsx3.
8. Target memory lasts 1 s for misses or 4 s under the hand. Sustained overlap estimates arrival, not physical touch. Phone and band stay connected to the laptop; the page is served locally.
