# shared

Contracts (`contracts.py`) and config loading (`config.py`) used by every module.

1. `cp shared/config.example.yaml config.yaml` and fill in your values.
2. `from shared.config import load_config` / `from shared.contracts import Detection, ...`
3. Tests: `pytest shared`

Change these files only through a `feat/shared` PR to `main`.
