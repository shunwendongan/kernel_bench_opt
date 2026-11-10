"""A rebuildable local identity index; never traverses model or dataset payloads."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

from .core import Sandbox, SandboxError, atomic_json, read_json, safe_id, utc_now

KINDS = {"task", "candidate", "run", "asset", "archive"}
RUN_STATUSES = {"planned", "completed", "failed", "cancelled"}
LEVELS = {"L1", "L2", "L3", "profile"}


def _directories(parent: Path) -> list[Path]:
    if parent.is_symlink() or not parent.is_dir():
        raise SandboxError(f"Expected an initialized directory: {parent.name}")
    children = sorted(parent.iterdir(), key=lambda item: item.name)
    for item in children:
        if item.is_symlink() or not item.is_dir():
            raise SandboxError(f"Unexpected non-directory or symlink in identity collection: {item.name}")
        safe_id(item.name)
    return children


def _schema(record: dict, label: str) -> None:
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise SandboxError(f"Unsupported schema in {label}")


def _tags(record: dict, label: str) -> list[str]:
    value = record.get("tags", [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SandboxError(f"tags must be a list of strings in {label}")
    return value


def _entry(kind: str, identity: str, task_id: str, record: dict, path: Path,
           sandbox: Sandbox, tags: list[str], candidate_id=None, run_id=None) -> dict:
    return {"kind": kind, "id": identity, "task_id": task_id,
            "candidate_id": candidate_id, "run_id": run_id,
            "status": record.get("status", record.get("verification") if kind == "asset" else None),
            "level": record.get("level"), "tags": tags,
            "record_path": path.relative_to(sandbox.root).as_posix(),
            "title": record.get("title") or identity}


def _archive_files(directory: Path, manifest: dict) -> None:
    """Check manifest structure and file presence, not file contents or large assets."""
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SandboxError("Archive manifest requires files")
    seen, total = set(), 0
    for entry in files:
        if not isinstance(entry, dict):
            raise SandboxError("Malformed archive file entry")
        value = entry.get("published_path")
        if not isinstance(value, str) or "\\" in value:
            raise SandboxError("Malformed archive-relative file path")
        parts = value.split("/")
        if len(parts) < 2 or parts[0] not in {"metadata", "files"} or any(p in {"", ".", ".."} or ":" in p for p in parts):
            raise SandboxError("Unsafe archive-relative file path")
        if value in seen:
            raise SandboxError("Duplicate archive file entry")
        seen.add(value)
        current = directory
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise SandboxError("Archive file path contains a symlink")
        if not current.is_file():
            raise SandboxError(f"Archived file is missing: {value}")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or current.stat().st_size != size:
            raise SandboxError(f"Archived file size differs from manifest: {value}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SandboxError("Archive file SHA256 is malformed")
        total += size
    if manifest.get("file_count") != len(files) or manifest.get("total_bytes") != total:
        raise SandboxError("Archive file counts/totals differ from entries")


def rebuild_index(sandbox: Sandbox) -> dict:
    """Rebuild atomically; a malformed source record leaves the previous cache intact."""
    sandbox.require_workspace()
    entries, tasks, candidates, runs = [], {}, {}, {}
    for directory in _directories(sandbox.root / "tasks"):
        task_id = directory.name
        task_dir = sandbox.task_path(task_id)
        task = read_json(task_dir / "task.json")
        _schema(task, "task")
        tags = _tags(task, "task")
        tasks[task_id] = task
        entries.append(_entry("task", task_id, task_id, task, task_dir / "task.json", sandbox, tags))
        inventory = read_json(task_dir / "assets.json")
        _schema(inventory, "asset inventory")
        if inventory.get("task_id") != task_id or not isinstance(inventory.get("assets"), list):
            raise SandboxError("Asset inventory identity/list is invalid")
        asset_ids = set()
        for asset in inventory["assets"]:
            if not isinstance(asset, dict):
                raise SandboxError("Asset inventory contains a non-object entry")
            asset_id = safe_id(asset.get("asset_id"))
            if asset_id in asset_ids:
                raise SandboxError(f"Duplicate asset ID: {asset_id}")
            asset_ids.add(asset_id)
            entries.append(_entry("asset", asset_id, task_id, asset, task_dir / "assets.json", sandbox, tags))
        local_candidates = {}
        for candidate_dir in _directories(task_dir / "candidates"):
            candidate_id = candidate_dir.name
            path = sandbox.candidate_path(task_id, candidate_id) / "candidate.json"
            candidate = read_json(path)
            _schema(candidate, "candidate")
            local_candidates[candidate_id] = candidate
            candidates[(task_id, candidate_id)] = candidate
            entries.append(_entry("candidate", candidate_id, task_id, candidate, path, sandbox, tags, candidate_id))
        for candidate_id, candidate in local_candidates.items():
            parent = candidate.get("parent_candidate_id")
            if parent is not None:
                safe_id(parent)
                if parent not in local_candidates:
                    raise SandboxError(f"Missing parent candidate for {candidate_id}")
        for candidate_id in local_candidates:
            chain, current = set(), candidate_id
            while current is not None:
                if current in chain:
                    raise SandboxError("Candidate parent references form a cycle")
                chain.add(current)
                current = local_candidates[current].get("parent_candidate_id")
        for run_dir in _directories(task_dir / "runs"):
            run_id = run_dir.name
            path = sandbox.run_path(task_id, run_id) / "run.json"
            run = read_json(path)
            _schema(run, "run")
            candidate_id = safe_id(run.get("candidate_id"))
            if candidate_id not in local_candidates:
                raise SandboxError(f"Missing candidate for run: {run_id}")
            if run.get("status") not in RUN_STATUSES or run.get("level") not in LEVELS:
                raise SandboxError(f"Invalid run status/level: {run_id}")
            snapshot = run.get("candidate_snapshot")
            if not isinstance(snapshot, dict) or snapshot.get("task_id") != task_id or snapshot.get("candidate_id") != candidate_id:
                raise SandboxError(f"Invalid candidate snapshot for run: {run_id}")
            asset_snapshots = run.get("asset_snapshots")
            if not isinstance(asset_snapshots, list):
                raise SandboxError(f"Invalid asset snapshots for run: {run_id}")
            used_ids = set()
            for asset in asset_snapshots:
                if not isinstance(asset, dict):
                    raise SandboxError("Run asset snapshot is not an object")
                identity = safe_id(asset.get("asset_id"))
                if identity not in asset_ids or identity in used_ids:
                    raise SandboxError(f"Missing or duplicate asset snapshot: {identity}")
                used_ids.add(identity)
            runs[(task_id, run_id)] = run
            entries.append(_entry("run", run_id, task_id, run, path, sandbox, tags, candidate_id, run_id))

    for task_dir in _directories(sandbox.root / "archives"):
        task_id = task_dir.name
        if task_id not in tasks:
            raise SandboxError(f"Archive references a missing task: {task_id}")
        for run_dir in _directories(task_dir):
            run_id = run_dir.name
            manifest = read_json(run_dir / "manifest.json")
            _schema(manifest, "archive")
            candidate_id = safe_id(manifest.get("candidate_id"))
            if manifest.get("kind") != "archived_run" or manifest.get("task_id") != task_id or manifest.get("run_id") != run_id:
                raise SandboxError("Archive identity does not match its directory")
            source = runs.get((task_id, run_id))
            if source is None or source["candidate_id"] != candidate_id:
                raise SandboxError("Archive references a missing or different run/candidate")
            if manifest.get("status") not in RUN_STATUSES - {"planned"} or manifest.get("level") not in LEVELS:
                raise SandboxError("Archive status/level is invalid")
            if manifest["status"] != source["status"] or manifest["level"] != source["level"]:
                raise SandboxError("Archive status/level differs from its source run")
            expected_source = f"tasks/{task_id}/runs/{run_id}/run.json"
            if manifest.get("source_record") != expected_source:
                raise SandboxError("Archive source_record is invalid")
            _archive_files(run_dir, manifest)
            entries.append(_entry("archive", run_id, task_id, manifest, run_dir / "manifest.json",
                                  sandbox, _tags(tasks[task_id], "task"), candidate_id, run_id))
    document = {"schema_version": 1, "updated_utc": utc_now(), "entries": entries,
                "semantics": "Derived identity cache; file hashes are recorded, not revalidated by indexing"}
    atomic_json(sandbox.root / "index.json", document)
    return document


def load_index(sandbox: Sandbox) -> dict:
    sandbox.require_workspace()
    path = sandbox.root / "index.json"
    if not path.exists() and not path.is_symlink():
        raise SandboxError("Index is missing; run index-rebuild")
    document = read_json(path)
    _schema(document, "index; run index-rebuild")
    if not isinstance(document.get("entries"), list):
        raise SandboxError("Malformed index; run index-rebuild")
    for entry in document["entries"]:
        if not isinstance(entry, dict) or entry.get("kind") not in KINDS:
            raise SandboxError("Malformed index entry; run index-rebuild")
        path = entry.get("record_path")
        if not isinstance(path, str) or "\\" in path or ":" in path or PurePosixPath(path).is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
            raise SandboxError("Unsafe indexed path; run index-rebuild")
    return document


def query_index(sandbox: Sandbox, kind=None, task_id=None, candidate_id=None,
                status=None, level=None, tag=None) -> list:
    if kind is not None and kind not in KINDS:
        raise SandboxError("Unknown index kind")
    for identity in (task_id, candidate_id):
        if identity is not None:
            safe_id(identity)
    filters = {"kind": kind, "task_id": task_id, "candidate_id": candidate_id,
               "status": status, "level": level}
    return [entry for entry in load_index(sandbox)["entries"]
            if all(value is None or entry.get(key) == value for key, value in filters.items())
            and (tag is None or tag in entry.get("tags", []))]
