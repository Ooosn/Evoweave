"""Shared TexVerse archive/source discovery helpers.

TexVerse outer zips often contain textures plus a second archive under
``source/``.  Pass0 and Pass1 must expand those nested archives in exactly the
same way, otherwise Pass0 can audit a source file that Pass1 cannot reconstruct.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


IMPORT_SUFFIX_ORDER = {".fbx": 0, ".glb": 1, ".gltf": 2, ".dae": 3, ".blend": 4}
IMPORTABLE_SUFFIXES = set(IMPORT_SUFFIX_ORDER)
NESTED_ARCHIVE_SUFFIXES = (".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz")


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        root = extract_dir.resolve()
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(extract_dir)


def is_nested_archive(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in NESTED_ARCHIVE_SUFFIXES)


def nested_archive_tool() -> str | None:
    explicit = os.environ.get("EVOWEAVE_7Z")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    return shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")


def nested_extract_dir(archive: Path) -> Path:
    safe_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", archive.name)
    return archive.parent / f"{safe_name}__unpacked"


def expand_nested_archives(
    extract_dir: Path,
    *,
    no_expand: bool = False,
    max_depth: int = 2,
    max_archives: int = 12,
    max_archive_mb: float = 0.0,
    timeout_sec: int = 120,
) -> list[dict]:
    """Expand nested source archives in-place and return audit records."""
    if no_expand:
        return []
    tool = nested_archive_tool()
    records: list[dict] = []
    extracted: set[Path] = set()
    max_depth = max(0, int(max_depth))
    max_archives = max(0, int(max_archives))
    max_bytes = int(max_archive_mb * 1024 * 1024) if max_archive_mb > 0 else 0

    for depth in range(max_depth):
        archives = [
            path
            for path in extract_dir.rglob("*")
            if path.is_file() and is_nested_archive(path) and path.resolve() not in extracted
        ]
        archives.sort(key=lambda path: (len(path.parts), str(path)))
        if max_archives > 0:
            remaining = max_archives - len(records)
            if remaining <= 0:
                break
            archives = archives[:remaining]
        if not archives:
            break

        for archive in archives:
            record = {
                "path": str(archive.relative_to(extract_dir)),
                "depth": depth,
                "size_bytes": archive.stat().st_size,
                "status": "pending",
            }
            extracted.add(archive.resolve())
            if max_bytes and archive.stat().st_size > max_bytes:
                record["status"] = "skipped_too_large"
                records.append(record)
                continue
            if tool is None:
                record["status"] = "skipped_no_7z"
                records.append(record)
                continue

            out_dir = nested_extract_dir(archive)
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                tool,
                "x",
                "-y",
                "-bd",
                "-bso0",
                "-bsp0",
                f"-o{out_dir}",
                str(archive),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                shutil.rmtree(out_dir, ignore_errors=True)
            else:
                record["status"] = "ok" if proc.returncode == 0 else "error"
                record["returncode"] = proc.returncode
                if proc.returncode != 0:
                    record["output_tail"] = proc.stdout.splitlines()[-20:]
                    shutil.rmtree(out_dir, ignore_errors=True)
                else:
                    record["output_dir"] = str(out_dir.relative_to(extract_dir))
            records.append(record)
    return records


def find_import_candidates(extract_dir: Path, max_candidates: int) -> tuple[list[Path], int]:
    files = [
        path
        for path in extract_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMPORT_SUFFIX_ORDER
        and not any(part.lower() == "__macosx" for part in path.parts)
        and not path.name.startswith("._")
    ]

    def candidate_key(path: Path) -> tuple[int, int, int, int, int, str]:
        lower_path = str(path).lower()
        lower_name = path.name.lower()
        compact_name = re.sub(r"[^a-z0-9]+", "", lower_name)
        explicitly_unskinned = any(
            marker in compact_name
            for marker in ("withoutskin", "noskin", "unskinned")
        )
        explicitly_skinned = not explicitly_unskinned and any(
            marker in compact_name
            for marker in ("withskin", "skinned")
        )
        skin_score = 100 if explicitly_skinned else (-100 if explicitly_unskinned else 0)
        animation_score = 0
        if "all animations" in lower_name or "all_animations" in lower_name or "all-animations" in lower_name:
            animation_score += 100
        if "anim" in lower_name or "animation" in lower_path:
            animation_score += 20
        if any(part.lower() == "animations" for part in path.parts):
            animation_score += 5
        if re.search(r"(^|[_\-\s])(mdl|model|mesh)([_\-\s.]|$)", lower_name):
            animation_score -= 20
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        return (
            IMPORT_SUFFIX_ORDER[path.suffix.lower()],
            -skin_score,
            -animation_score,
            -size,
            len(path.parts),
            str(path),
        )

    files.sort(key=candidate_key)
    if max_candidates <= 0:
        return files, len(files)
    return files[:max_candidates], len(files)


def relative_source_from_candidate(asset_id: str, candidate_path: str | None) -> str | None:
    if not candidate_path:
        return None
    parts = Path(candidate_path).parts
    if asset_id not in parts:
        return None
    idx = parts.index(asset_id)
    rel_parts = parts[idx + 1 :]
    if not rel_parts:
        return None
    return str(Path(*rel_parts))
