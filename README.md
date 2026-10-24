# Octaby

Wearable that guides a blind user's hand to an object. Read `AGENTS.md` first.

```
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp shared/config.example.yaml config.yaml                # then edit
pytest
python -m app.main --mock-band
```

PRD: `docs/Octaby v2 Product Requirements.pdf`
