#!/usr/bin/env python3
"""Apply this experiment to a dedicated checkout of the pinned public MNN source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]
PATCHES = REPO / "scripts/patches"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def prepare(source, dry_run=False):
    source = source.resolve()
    manifest = json.loads((PATCHES / "manifest.json").read_text())
    if (source / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != manifest["mnn_commit"]:
            raise RuntimeError(f"MNN HEAD is {actual}; expected {manifest['mnn_commit']}")
    changes = []
    # Validate every destination before changing any file. A previously prepared
    # tree is accepted; unrelated local edits are never silently overwritten.
    for item in manifest["patches"]:
        target = source / item["path"]
        actual = digest(target)
        if actual == item["after_sha256"]:
            continue
        if actual != item["before_sha256"]:
            raise RuntimeError(f"Unexpected source content: {target}; use a dedicated pinned checkout")
        patch = PATCHES / item["patch"]
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        changes.append(("patch", item["path"], patch, item["after_sha256"]))
    for item in manifest["overlays"]:
        origin, target = REPO / item["source"], source / item["path"]
        if not origin.is_file():
            raise RuntimeError(f"Missing repository source: {origin}")
        expected, actual = digest(origin), digest(target)
        if actual == expected:
            continue
        # A tree prepared with an earlier candidate of this repository may be upgraded.
        if actual != item["before_sha256"] and actual not in item.get("accepted_previous_sha256", []):
            raise RuntimeError(f"Unexpected source content: {target}; use a dedicated pinned checkout")
        changes.append(("copy", item["path"], origin, expected))
    for action, relative, origin, expected in changes:
        print(f"{action}: {relative}", flush=True)
        if dry_run:
            continue
        target = source / relative
        if action == "patch":
            subprocess.run(["git", "apply", str(origin)], cwd=source, check=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        if digest(target) != expected:
            raise RuntimeError(f"Prepared content hash mismatch: {target}")
    print("PREPARE_DRY_RUN_OK" if dry_run else "PREPARE_OK", flush=True)
    return manifest["mnn_commit"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnn-source", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare(args.mnn_source, args.dry_run)


if __name__ == "__main__":
    main()
