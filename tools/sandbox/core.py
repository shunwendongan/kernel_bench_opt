"""Local identity and snapshot storage; never execute a recorded workload."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit
import uuid


class SandboxError(ValueError):
    """An invalid request or inconsistent local workspace."""


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_HASH = re.compile(r"[0-9a-fA-F]{64}\Z")
_COMMIT = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")
_LEVELS = {"L1", "L2", "L3", "profile"}
_ASSET_KINDS = {"model", "dataset", "capture", "profile", "report", "other"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise SandboxError("IDs require 1–96 ASCII letters/digits/._-, beginning with a letter or digit")
    return value


def _absolute_no_symlinks(path: Path) -> Path:
    # Check lexical parents before resolve(), which would erase symlink evidence.
    path = Path(os.path.abspath(Path(path).expanduser()))
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise SandboxError(f"Symlinks are not allowed: {component}")
    return path


def _regular(path: Path) -> Path:
    path = _absolute_no_symlinks(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise SandboxError(f"Expected a regular file: {path}")
    return path


def _invalid_constant(value: str):
    raise SandboxError(f"Non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SandboxError("JSON numeric value exceeds the finite floating-point range")
    return number


def read_json(path: Path) -> dict:
    path = _regular(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_invalid_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise SandboxError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path = _absolute_no_symlinks(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and not path.is_file():
        raise SandboxError(f"Expected a regular JSON destination: {path}")
    if not path.parent.is_dir():
        raise SandboxError(f"Missing destination directory: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # A cooperative CLI writer holds the workspace lock. Recheck the target
        # so an obvious symlink replacement is not followed or overwritten.
        _absolute_no_symlinks(path)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    path = _regular(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise SandboxError("Asset changed while its identity was being read")
    return digest.hexdigest()


def _text(value: str | None, name: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise SandboxError(f"{name} must be {'a non-empty' if required else 'a'} string")
    return value


def _commit(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not _COMMIT.fullmatch(value)):
        raise SandboxError("source_commit must be a full 40- or 64-character hexadecimal commit identity")
    return value.lower() if value else None


def _schema(record: dict, name: str) -> None:
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise SandboxError(f"Unsupported schema_version in {name}")


@contextmanager
def _new_directory(target: Path):
    """Prepare a private sibling tree and publish it only after all writes finish."""
    target = _absolute_no_symlinks(target)
    if target.exists():
        raise SandboxError(f"Refusing to overwrite existing directory: {target}")
    if not target.parent.is_dir():
        raise SandboxError(f"Missing parent directory: {target.parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.pending-", dir=target.parent))
    try:
        yield temporary
        if target.exists() or target.is_symlink():
            raise SandboxError(f"Destination appeared during creation: {target}")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class Sandbox:
    def __init__(self, root: Path, config: dict):
        self.root = _absolute_no_symlinks(Path(root)).resolve()
        self.config = deepcopy(config)

    def require_workspace(self) -> dict:
        _absolute_no_symlinks(self.root)
        record = read_json(self.root / "workspace.json")
        _schema(record, "workspace.json")
        if record.get("record_type") != "sandbox_workspace":
            raise SandboxError("The directory is not a task sandbox workspace")
        for name in ("tasks", "archives"):
            path = _absolute_no_symlinks(self.root / name)
            if not path.is_dir():
                raise SandboxError(f"Missing workspace directory: {name}")
        return record

    def init_workspace(self) -> dict:
        # Initialization has no workspace lock yet; a sibling O_EXCL reservation
        # serializes cooperating initializers without touching an existing root.
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _absolute_no_symlinks(self.root.parent)
        reservation = self.root.parent / f".{self.root.name}.init.lock"
        try:
            descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise SandboxError("Workspace initialization is already reserved; inspect the local init lock") from exc
        identity = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}) + "\n")
            if self.root.exists():
                return self.require_workspace()
            record = {
                "schema_version": 1,
                "record_type": "sandbox_workspace",
                "workspace_id": str(uuid.uuid4()),
                "created_at": utc_now(),
                "configuration_at_creation": deepcopy(self.config),
                "execution_policy": "metadata_only_no_workload_execution",
            }
            with _new_directory(self.root) as temporary:
                (temporary / "tasks").mkdir()
                (temporary / "archives").mkdir()
                atomic_json(temporary / "workspace.json", record)
            return record
        finally:
            self._remove_owned_lock(reservation, identity)

    @staticmethod
    def _remove_owned_lock(path: Path, identity) -> None:
        try:
            current = path.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                path.unlink()
        except FileNotFoundError:
            pass

    @contextmanager
    def lock(self):
        self.require_workspace()
        path = self.root / ".write.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise SandboxError("Workspace write lock exists; wait for its owner or inspect a stale lock manually") from exc
        identity = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}) + "\n")
            yield
        finally:
            self._remove_owned_lock(path, identity)

    def _identity_path(self, path: Path, filename: str, expected: dict) -> Path:
        path = _absolute_no_symlinks(path)
        if not path.is_dir():
            raise SandboxError(f"Unknown record directory: {path}")
        value = read_json(path / filename)
        _schema(value, filename)
        if any(value.get(key) != wanted for key, wanted in expected.items()):
            raise SandboxError(f"Stored identity does not match its directory: {path}")
        return path

    def task_path(self, task_id: str) -> Path:
        self.require_workspace()
        return self._identity_path(self.root / "tasks" / safe_id(task_id), "task.json", {"task_id": task_id})

    def candidate_path(self, task_id: str, candidate_id: str) -> Path:
        task = self.task_path(task_id)
        return self._identity_path(task / "candidates" / safe_id(candidate_id), "candidate.json",
                                   {"task_id": task_id, "candidate_id": candidate_id})

    def run_path(self, task_id: str, run_id: str) -> Path:
        task = self.task_path(task_id)
        path = self._identity_path(task / "runs" / safe_id(run_id), "run.json", {"task_id": task_id, "run_id": run_id})
        record = read_json(path / "run.json")
        self.candidate_path(task_id, record.get("candidate_id"))
        return path

    def task_init(self, task_id: str, title: str, objective: str, model: str | None = None,
                  dataset: str | None = None, branch: str | None = None, tags: list[str] | None = None) -> dict:
        self.require_workspace()
        task_id = safe_id(task_id)
        tags = [] if tags is None else tags
        if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise SandboxError("tags must be a list of non-empty strings")
        record = {
            "schema_version": 1, "record_type": "task", "task_id": task_id,
            "title": _text(title, "title", required=True),
            "objective": _text(objective, "objective", required=True),
            "model": _text(model, "model"), "dataset": _text(dataset, "dataset"),
            "branch": _text(branch, "branch"), "tags": list(dict.fromkeys(tags)),
            "status": "active", "created_at": utc_now(),
        }
        with _new_directory(self.root / "tasks" / task_id) as temporary:
            for name in ("baseline", "candidates", "runs"):
                (temporary / name).mkdir()
            atomic_json(temporary / "task.json", record)
            atomic_json(temporary / "assets.json", {"schema_version": 1, "task_id": task_id, "assets": []})
        return record

    def candidate_init(self, task_id: str, candidate_id: str, parent_id: str | None = None,
                       hypothesis: str = "", source_commit: str | None = None,
                       source_dirty: bool | None = None) -> dict:
        task = self.task_path(task_id)
        candidate_id = safe_id(candidate_id)
        if parent_id is not None:
            self.candidate_path(task_id, parent_id)
        if source_dirty is not None and type(source_dirty) is not bool:
            raise SandboxError("source_dirty must be boolean or null")
        record = {
            "schema_version": 1, "record_type": "candidate", "task_id": task_id,
            "candidate_id": candidate_id, "parent_candidate_id": parent_id,
            "status": "proposed", "hypothesis": _text(hypothesis, "hypothesis"),
            "source": {"git_commit": _commit(source_commit), "dirty": source_dirty},
            "created_at": utc_now(),
        }
        with _new_directory(task / "candidates" / candidate_id) as temporary:
            (temporary / "patch").mkdir()
            atomic_json(temporary / "candidate.json", record)
            (temporary / "design.md").write_text(
                f"# Candidate {candidate_id}\n\n"
                f"任务：`{task_id}`；父候选：`{parent_id or 'none'}`。\n\n"
                f"## 假设\n\n{hypothesis or '待填写：依据哪条观察，预计改变哪项成本。'}\n\n"
                "## 不变量与适用范围\n\n待填写：权重、精度、输入、布局、shape、回退条件。\n\n"
                "## 变更与证据\n\n待填写：实现差异、源码身份、相关profile与映射。\n\n"
                "## 评测与决策\n\n未运行。待填写：L1/L2/L3协议、原始结果、限制与决定。\n",
                encoding="utf-8",
            )
        return record

    def _assets(self, task_id: str) -> dict:
        value = read_json(self.task_path(task_id) / "assets.json")
        _schema(value, "assets.json")
        if value.get("task_id") != task_id or not isinstance(value.get("assets"), list):
            raise SandboxError("Asset manifest does not match the task")
        seen = set()
        for item in value["assets"]:
            if not isinstance(item, dict):
                raise SandboxError("Asset entries must be objects")
            asset_id = safe_id(item.get("asset_id"))
            if asset_id in seen:
                raise SandboxError("Asset manifest contains duplicate IDs")
            seen.add(asset_id)
        return value

    def asset_add(self, task_id: str, asset_id: str, kind: str, path: Path | None = None,
                  uri: str | None = None, sha256: str | None = None, size_bytes: int | None = None,
                  license: str | None = None, source: str | None = None) -> dict:
        task = self.task_path(task_id)
        asset_id = safe_id(asset_id)
        manifest = self._assets(task_id)
        if any(item["asset_id"] == asset_id for item in manifest["assets"]):
            raise SandboxError("Asset ID already exists; register a new identity for a new version")
        if kind not in _ASSET_KINDS:
            raise SandboxError("Unsupported asset kind")
        if (path is None) == (uri is None):
            raise SandboxError("Supply exactly one local path or external URI")
        if sha256 is not None and (not isinstance(sha256, str) or not _HASH.fullmatch(sha256)):
            raise SandboxError("sha256 must contain exactly 64 hexadecimal characters")
        if size_bytes is not None and (type(size_bytes) is not int or size_bytes < 0):
            raise SandboxError("size_bytes must be a non-negative integer")
        sha256 = sha256.lower() if sha256 is not None else None
        if path is not None:
            local = _regular(Path(path))
            before = local.stat()
            actual_hash = sha256_file(local)
            after = local.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise SandboxError("Asset changed while being registered")
            if sha256 is not None and sha256 != actual_hash:
                raise SandboxError("Local asset does not match the declared SHA256")
            if size_bytes is not None and size_bytes != after.st_size:
                raise SandboxError("Local asset does not match the declared size")
            location = {"type": "local_file", "path": str(local)}
            sha256, size_bytes = actual_hash, after.st_size
            verification = "local_file_hashed"
        else:
            _text(uri, "uri", required=True)
            parsed = urlsplit(uri)
            if not parsed.scheme or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise SandboxError("Use a stable URI with a scheme and no credentials, query or fragment")
            location = {"type": "external_reference", "uri": uri}
            verification = "declared_not_verified"
        record = {
            "schema_version": 1, "asset_id": asset_id, "kind": kind,
            "location": location, "sha256": sha256, "size_bytes": size_bytes,
            "verification": verification, "license": _text(license, "license"),
            "source": _text(source, "source"), "registered_at": utc_now(),
        }
        manifest["assets"].append(record)
        atomic_json(task / "assets.json", manifest)
        return record

    def run_init(self, task_id: str, candidate_id: str, run_id: str, level: str,
                 asset_ids: list[str] | None = None, device_id: str | None = None,
                 source_commit: str | None = None) -> dict:
        task = self.task_path(task_id)
        candidate = read_json(self.candidate_path(task_id, candidate_id) / "candidate.json")
        run_id = safe_id(run_id)
        if level not in _LEVELS:
            raise SandboxError("level must be L1, L2, L3 or profile")
        asset_ids = [] if asset_ids is None else asset_ids
        if not isinstance(asset_ids, list):
            raise SandboxError("asset_ids must be a list of distinct IDs")
        for asset_id in asset_ids:
            safe_id(asset_id)
        if len(set(asset_ids)) != len(asset_ids):
            raise SandboxError("asset_ids must be a list of distinct IDs")
        assets = {item["asset_id"]: item for item in self._assets(task_id)["assets"]}
        snapshots = []
        for asset_id in asset_ids:
            safe_id(asset_id)
            if asset_id not in assets:
                raise SandboxError(f"Unknown task asset: {asset_id}")
            snapshots.append(deepcopy(assets[asset_id]))
        source_commit = _commit(source_commit)
        candidate_source = candidate.get("source", {})
        if not isinstance(candidate_source, dict):
            raise SandboxError("Candidate source must be an object")
        candidate_commit = _commit(candidate_source.get("git_commit"))
        if source_commit is not None and candidate_commit is not None and source_commit != candidate_commit:
            raise SandboxError("Run source commit conflicts with the candidate; create a new candidate")
        record = {
            "schema_version": 1, "record_type": "planned_run", "task_id": task_id,
            "candidate_id": candidate_id, "run_id": run_id, "status": "planned", "level": level,
            "created_at": utc_now(), "finished_at": None,
            "candidate_snapshot": deepcopy(candidate), "asset_snapshots": snapshots,
            "source": {"git_commit": source_commit or candidate_commit, "dirty": candidate_source.get("dirty")},
            "device_id": _text(device_id, "device_id"), "result": None,
        }
        with _new_directory(task / "runs" / run_id) as temporary:
            for name in ("raw", "analysis", "benchmark", "artifacts"):
                (temporary / name).mkdir()
            atomic_json(temporary / "run.json", record)
        return record

    def run_finish(self, task_id: str, run_id: str, status: str,
                   result_path: Path | None = None, note: str = "") -> dict:
        path = self.run_path(task_id, run_id)
        record = read_json(path / "run.json")
        if record.get("status") != "planned":
            raise SandboxError("Only planned runs can transition to a terminal status")
        if status not in {"completed", "failed", "cancelled"}:
            raise SandboxError("Unsupported terminal run status")
        if (self.root / "archives" / safe_id(task_id) / safe_id(run_id)).exists():
            raise SandboxError("An archived run cannot be changed through this interface")
        payload = read_json(result_path) if result_path is not None else None
        if status == "completed":
            if payload is None:
                raise SandboxError("completed requires an explicit result JSON object")
            if payload.get("record_type") == "template_not_measurement" or payload.get("evidence_kind") == "not_run":
                raise SandboxError("An unexecuted result template cannot complete a run")
        record.update({
            "record_type": "run_record", "status": status, "finished_at": utc_now(),
            "result": {"source": "caller_declared", "payload": payload,
                       "note": _text(note, "note"),
                       "correctness_inferred_from_status": False},
        })
        atomic_json(path / "run.json", record)
        return record
