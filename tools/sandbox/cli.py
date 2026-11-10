"""Local task records only: no inference, subprocess execution, network, or git writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import Sandbox, SandboxError, read_json, safe_id
from .archive import archive_run
from .index import rebuild_index, query_index

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY / "configs" / "sandbox.default.json"


def config_from(path: Path) -> dict:
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise SandboxError("Unsupported sandbox configuration schema_version")
    archive = config.get("archive")
    if not isinstance(archive, dict):
        raise SandboxError("Configuration requires an archive object")
    for name in ("max_file_bytes", "max_total_bytes"):
        value = archive.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SandboxError(f"archive.{name} must be a positive integer")
    suffixes = archive.get("allowed_suffixes")
    if not isinstance(suffixes, list) or not suffixes or any(
        not isinstance(s, str) or not s.startswith(".") or s != s.lower()
        for s in suffixes
    ):
        raise SandboxError("archive.allowed_suffixes must contain lowercase suffixes beginning with '.'")
    root = config.get("workspace_root", ".sandbox")
    if not isinstance(root, str) or not root.strip():
        raise SandboxError("workspace_root must be a non-empty path string")
    return config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, help="Workspace directory; relative to the calling directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize an empty local workspace")

    task = commands.add_parser("task-init", help="Create Task identity and directories")
    task.add_argument("--task", required=True)
    task.add_argument("--title", required=True)
    task.add_argument("--objective", required=True)
    task.add_argument("--model")
    task.add_argument("--dataset")
    task.add_argument("--branch")
    task.add_argument("--tag", action="append", default=[])

    asset = commands.add_parser("asset-add", help="Register one file or external reference; never fetch it")
    asset.add_argument("--task", required=True)
    asset.add_argument("--asset", required=True)
    asset.add_argument("--kind", choices=("model", "dataset", "capture", "profile", "report", "other"), required=True)
    location = asset.add_mutually_exclusive_group(required=True)
    location.add_argument("--path", type=Path, help="Hash a local regular file; directories are not scanned")
    location.add_argument("--uri", help="Record an external asset identity without accessing it")
    asset.add_argument("--sha256", help="Expected hash for a local file, or declared hash for an external reference")
    asset.add_argument("--size-bytes", type=int)
    asset.add_argument("--license")
    asset.add_argument("--source", help="Public provenance/version reference")

    candidate = commands.add_parser("candidate-init", help="Create a candidate with optional parent")
    candidate.add_argument("--task", required=True)
    candidate.add_argument("--candidate", required=True)
    candidate.add_argument("--parent")
    candidate.add_argument("--hypothesis", default="")
    candidate.add_argument("--source-commit")
    candidate.add_argument("--source-dirty", action=argparse.BooleanOptionalAction, default=None)

    run = commands.add_parser("run-init", help="Prepare directories and freeze input identities; does not execute a command")
    run.add_argument("--task", required=True)
    run.add_argument("--candidate", required=True)
    run.add_argument("--run", required=True)
    run.add_argument("--level", choices=("L1", "L2", "L3", "profile"), required=True)
    run.add_argument("--asset", action="append", default=[], help="Task asset ID to snapshot; repeat for each input")
    run.add_argument("--device-id")
    run.add_argument("--source-commit")

    finish = commands.add_parser("run-finish", help="Record a caller-declared terminal result, without measuring it")
    finish.add_argument("--task", required=True)
    finish.add_argument("--run", required=True)
    finish.add_argument("--status", choices=("completed", "failed", "cancelled"), required=True)
    finish.add_argument("--result", type=Path, help="JSON object; required for completed")
    finish.add_argument("--note", default="")

    archive = commands.add_parser("archive-run", help="Seal an explicit selection of completed run files")
    archive.add_argument("--task", required=True)
    archive.add_argument("--run", required=True)
    archive.add_argument("--include", action="append", required=True, help="One regular file relative to the run; no glob or directory scan")

    commands.add_parser("index-rebuild", help="Rebuild the derived index from stored identities and archive manifests")
    listing = commands.add_parser("list", help="Query the derived local index")
    listing.add_argument("--kind", choices=("task", "candidate", "run", "asset", "archive"))
    listing.add_argument("--task")
    listing.add_argument("--candidate")
    listing.add_argument("--status")
    listing.add_argument("--level", choices=("L1", "L2", "L3", "profile"))
    listing.add_argument("--tag")

    show = commands.add_parser("show", help="Read one stored identity without executing anything")
    show.add_argument("--kind", choices=("task", "candidate", "run", "asset", "archive"), required=True)
    show.add_argument("--task", required=True)
    show.add_argument("--id", help="Required except when kind is task")
    return p


def show_record(sandbox: Sandbox, kind: str, task_id: str, record_id: str | None) -> dict:
    task = sandbox.task_path(task_id)
    if kind == "task":
        return read_json(task / "task.json")
    if record_id is None:
        raise SandboxError("show requires --id for candidate, run, asset or archive")
    safe_id(record_id)
    if kind == "candidate":
        return read_json(sandbox.candidate_path(task_id, record_id) / "candidate.json")
    if kind == "run":
        return read_json(sandbox.run_path(task_id, record_id) / "run.json")
    if kind == "asset":
        assets = read_json(task / "assets.json")
        for item in assets.get("assets", []):
            if item.get("asset_id") == record_id:
                return item
        raise SandboxError(f"Unknown asset: {record_id}")
    sandbox.run_path(task_id, record_id)
    path = sandbox.root / "archives" / safe_id(task_id) / record_id / "manifest.json"
    # read_json also rejects symlinks. Check every archive component before reading.
    for part in (path.parent.parent.parent, path.parent.parent, path.parent, path):
        if part.is_symlink():
            raise SandboxError("Archive path must not contain symlinks")
    value = read_json(path)
    if value.get("task_id") != task_id or value.get("run_id") != record_id:
        raise SandboxError("Archive identity does not match its path")
    return value


def mutate(sandbox: Sandbox, args: argparse.Namespace) -> dict:
    if args.command == "task-init":
        return sandbox.task_init(args.task, args.title, args.objective, model=args.model,
                                 dataset=args.dataset, branch=args.branch, tags=args.tag)
    if args.command == "asset-add":
        return sandbox.asset_add(args.task, args.asset, args.kind, path=args.path, uri=args.uri,
                                 sha256=args.sha256, size_bytes=args.size_bytes,
                                 license=args.license, source=args.source)
    if args.command == "candidate-init":
        return sandbox.candidate_init(args.task, args.candidate, parent_id=args.parent,
                                      hypothesis=args.hypothesis, source_commit=args.source_commit,
                                      source_dirty=args.source_dirty)
    if args.command == "run-init":
        return sandbox.run_init(args.task, args.candidate, args.run, args.level,
                                asset_ids=args.asset, device_id=args.device_id,
                                source_commit=args.source_commit)
    if args.command == "run-finish":
        return sandbox.run_finish(args.task, args.run, args.status,
                                  result_path=args.result, note=args.note)
    if args.command == "archive-run":
        return archive_run(sandbox, args.task, args.run, args.include)
    if args.command == "index-rebuild":
        return rebuild_index(sandbox)
    raise SandboxError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = config_from(args.config.expanduser())
        root = args.root if args.root is not None else Path(config.get("workspace_root", ".sandbox"))
        sandbox = Sandbox(root.expanduser(), config)
        if args.command == "init":
            result = sandbox.init_workspace()
            with sandbox.lock():
                rebuild_index(sandbox)
        else:
            sandbox.require_workspace()
            if args.command == "list":
                result = {"entries": query_index(sandbox, kind=args.kind, task_id=args.task,
                            candidate_id=args.candidate, status=args.status,
                            level=args.level, tag=args.tag)}
            elif args.command == "show":
                result = show_record(sandbox, args.kind, args.task, args.id)
            else:
                with sandbox.lock():
                    result = mutate(sandbox, args)
                    if args.command != "index-rebuild":
                        try:
                            rebuild_index(sandbox)
                        except (SandboxError, OSError, ValueError) as exc:
                            # The primary record is already committed; do not pretend it rolled back.
                            print(json.dumps({"ok": False, "record_saved": True, "result": result,
                                              "error": f"Index refresh failed: {exc}",
                                              "next_step": "Resolve the index error, then run index-rebuild"},
                                             ensure_ascii=False, indent=2), file=sys.stderr)
                            return 2
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (SandboxError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
