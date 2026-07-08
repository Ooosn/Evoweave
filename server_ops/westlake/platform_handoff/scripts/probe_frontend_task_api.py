from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "frontend_probe"
TARGETS = [
    "CreateTask.8634abc1.js",
    "2537.f3ca8e5f.js",
    "3701.a00c4a46.js",
    "6392.cc0aff10.js",
    "1929.baac9071.js",
    "TaskList.5f15baa1.js",
]

PATTERNS = [
    r"api/[A-Za-z0-9_./{}?=&:-]+",
    r'url:"[^"]+',
    r"url:'[^']+",
    r'"/[A-Za-z0-9_./{}?=&:-]*train[A-Za-z0-9_./{}?=&:-]*"',
    r'"/[A-Za-z0-9_./{}?=&:-]*task[A-Za-z0-9_./{}?=&:-]*"',
]

TERMS = [
    "submit",
    "createTask",
    "taskName",
    "startFile",
    "command",
    "execScript",
    "resourceGroup",
    "deployType",
]


def one_line(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ")


def main() -> None:
    for name in TARGETS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        print(f"FILE {name} len={len(text)}")
        for pattern in PATTERNS:
            values = sorted(set(re.findall(pattern, text)))
            if values:
                print(f" PAT {pattern}")
                for value in values[:100]:
                    print(f"  {value[:240]}")
        for term in TERMS:
            pos = 0
            count = 0
            while count < 6:
                idx = text.find(term, pos)
                if idx < 0:
                    break
                snippet = one_line(text[max(0, idx - 220) : idx + 360])
                print(f" SNIP {term} {snippet}")
                pos = idx + len(term)
                count += 1


if __name__ == "__main__":
    main()
