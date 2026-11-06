"""Explicit, bounded local archives. Callers hold Sandbox.lock() for writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile

from .core import Sandbox, SandboxError, read_json, safe_id, utc_now

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _regular(path: Path, root: Path) -> os.stat_result:
    """Check each component without following an archive-selected symlink."""
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SandboxError("Selected file must be inside its run") from exc
    current = root
    if current.is_symlink() or not current.is_dir():
        raise SandboxError("Run root must be a regular directory")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SandboxError(f"Symlink is not an archive input: {relative.as_posix()}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise SandboxError(f"Archive input must be one regular file: {relative.as_posix()}")
    return info


def _fingerprint(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _relative_file(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SandboxError("--include must be a non-empty, slash-separated relative file path")
    raw = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in raw):
        raise SandboxError(f"Unsafe archive-relative path: {value}")
    # Reject drive-letter forms on every platform, not just on Windows.
    if any(":" in part for part in raw):
        raise SandboxError(f"Unsafe archive-relative path: {value}")
    return Path(*PurePosixPath(value).parts)


def _copy_stable(source: Path, destination: Path, root: Path,
                 expected: os.stat_result) -> tuple[int, str]:
    """Copy and compare a second read; a producer must stop writing before archive."""
    if _fingerprint(_regular(source, root)) != _fingerprint(expected):
        raise SandboxError(f"Archive source changed before copy: {source.relative_to(root)}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        with os.fdopen(descriptor, "rb") as src:
            descriptor = -1
            if _fingerprint(os.fstat(src.fileno())) != _fingerprint(expected):
                raise SandboxError("Archive input changed while opening")
            digest = hashlib.sha256()
            size = 0
            with destination.open("xb") as dst:
                while chunk := src.read(1024 * 1024):
                    size += len(chunk)
                    if size > expected.st_size:
                        raise SandboxError("Archive input grew while copying")
                    digest.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            if size != expected.st_size:
                raise SandboxError("Archive input shrank while copying")
            # A second read catches content changes even when size stays fixed.
            src.seek(0)
            check = hashlib.sha256()
            while chunk := src.read(1024 * 1024):
                check.update(chunk)
            if check.digest() != digest.digest() or _fingerprint(os.fstat(src.fileno())) != _fingerprint(expected):
                raise SandboxError("Archive input changed during copying")
        if _fingerprint(_regular(source, root)) != _fingerprint(expected):
            raise SandboxError("Archive input path changed during copying")
        return size, digest.hexdigest()
    finally:
        if descriptor != -1:
            os.close(descriptor)


def archive_run(sandbox: Sandbox, task_id: str, run_id: str, includes: list[str]) -> dict:
    """Publish a new archive by rename; never overwrite archives or original runs."""
    sandbox.require_workspace()
    task_dir = sandbox.task_path(task_id)
    run_dir = sandbox.run_path(task_id, run_id)
    run = read_json(run_dir / "run.json")
    if run.get("status") not in TERMINAL_STATUSES:
        raise SandboxError("Only completed, failed, or cancelled runs can be archived")
    candidate_id = safe_id(run.get("candidate_id"))
    sandbox.candidate_path(task_id, candidate_id)
    snapshot = run.get("candidate_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("task_id") != task_id or snapshot.get("candidate_id") != candidate_id:
        raise SandboxError("Run lacks a valid candidate snapshot; current candidate is not a substitute")
    if not isinstance(includes, list) or not includes:
        raise SandboxError("Select at least one run-relative evidence file with --include")

    policy = sandbox.config["archive"]
    max_file = policy["max_file_bytes"]
    max_total = policy["max_total_bytes"]
    allowed = set(policy["allowed_suffixes"])
    selected = []
    seen = set()
    total = 0
    for value in includes:
        relative = _relative_file(value)
        key = relative.as_posix()
        if key in seen:
            raise SandboxError(f"Duplicate archive input: {key}")
        seen.add(key)
        if relative.suffix.lower() not in allowed:
            raise SandboxError(f"Archive suffix is not allowed: {key}")
        info = _regular(run_dir / relative, run_dir)
        if info.st_size > max_file:
            raise SandboxError(f"Archive input exceeds max_file_bytes: {key}")
        total += info.st_size
        selected.append((relative, info))

    task = read_json(task_dir / "task.json")
    metadata = [("task.json", task, "task.json"),
                ("candidate.json", snapshot, f"runs/{run_id}/run.json#/candidate_snapshot"),
                ("run.json", run, f"runs/{run_id}/run.json")]
    encoded_metadata = []
    for name, value, original in metadata:
        payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > max_file:
            raise SandboxError(f"Archive metadata exceeds max_file_bytes: {name}")
        total += len(payload)
        encoded_metadata.append((name, payload, original))
    if total > max_total:
        raise SandboxError("Selected evidence and metadata exceed max_total_bytes")

    archive_root = sandbox.root / "archives"
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise SandboxError("archives must be an initialized regular directory")
    parent = archive_root / safe_id(task_id)
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise SandboxError("Archive task path must be a regular directory")
    parent.mkdir(exist_ok=True)
    destination = parent / safe_id(run_id)
    if destination.exists() or destination.is_symlink():
        raise SandboxError("Archive already exists; use a new Run ID instead of overwriting it")
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=parent))
    try:
        (stage / "metadata").mkdir()
        (stage / "files").mkdir()
        entries = []
        for name, payload, original in encoded_metadata:
            target = Path("metadata") / name
            (stage / target).write_bytes(payload)
            entries.append({"original_relative_path": f"tasks/{task_id}/{original}",
                            "published_path": target.as_posix(), "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
        for relative, info in selected:
            target = Path("files") / relative
            (stage / target).parent.mkdir(parents=True, exist_ok=True)
            size, digest = _copy_stable(run_dir / relative, stage / target, run_dir, info)
            entries.append({"original_relative_path": (Path("tasks") / task_id / "runs" / run_id / relative).as_posix(),
                            "published_path": target.as_posix(), "size_bytes": size, "sha256": digest})
        # Freeze the same terminal metadata that was selected before copying evidence.
        if read_json(run_dir / "run.json") != run or read_json(task_dir / "task.json") != task:
            raise SandboxError("Task/run metadata changed while archiving")
        for relative, expected in selected:
            if _fingerprint(_regular(run_dir / relative, run_dir)) != _fingerprint(expected):
                raise SandboxError("An archive input changed before the snapshot was finalized")
        manifest = {"schema_version": 1, "kind": "archived_run", "task_id": task_id,
                    "candidate_id": candidate_id, "run_id": run_id, "status": run["status"],
                    "level": run.get("level"), "created_utc": utc_now(),
                    "source_record": (Path("tasks") / task_id / "runs" / run_id / "run.json").as_posix(),
                    "files": entries, "file_count": len(entries),
                    "total_bytes": sum(item["size_bytes"] for item in entries),
                    "archive_semantics": "API refuses overwrite; not an OS-immutable or signed archive",
                    "manifest_excluded_from_totals": True}
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise SandboxError("Archive destination appeared while preparing the snapshot")
        stage.rename(destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
