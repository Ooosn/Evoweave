from __future__ import annotations

from pathlib import Path


text = (Path(__file__).resolve().parents[1] / "frontend_probe" / "1929.baac9071.js").read_text(
    encoding="utf-8",
    errors="ignore",
)

terms = [
    "createTask(){",
    "api/iresource/v1/train",
    "taskCmd",
    "const t=",
]

for term in terms:
    idx = text.find(term)
    print(f"TERM {term!r} idx={idx}")
    print(text[max(0, idx - 1800) : idx + 2600])
    print("\n---\n")
