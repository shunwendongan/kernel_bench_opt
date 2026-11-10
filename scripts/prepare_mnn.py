#!/usr/bin/env python3
"""Apply this experiment to a dedicated checkout of the pinned public MNN source.

The engine overlay manifest (scripts/patches/manifest.json, schema 2) has three
layers: git patches against the pinned commit, common overlays shared by every
candidate set (src/<engine>/), and named overlay sets (one per deployable
candidate combination, sources under src/<engine>/overlays/<candidate-id>/ and
kernels/<candidate-id>/). Select a set with --candidate-set; the manifest's
default_overlay_set is used otherwise. Schema 1 manifests (a flat "overlays"
list) are still accepted.
"""
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


def load_manifest(path=PATCHES / "manifest.json"):
    return json.loads(path.read_text())


def resolve_overlays(manifest, overlay_set=None):
    """Return (set_name, overlay list) for the requested or default set."""
    if "overlay_sets" not in manifest:          # schema 1
        if overlay_set not in (None, "default"):
            raise RuntimeError("This manifest has no overlay sets; omit --candidate-set")
        return "default", list(manifest["overlays"])
    name = overlay_set or manifest.get("default_overlay_set")
    sets = manifest["overlay_sets"]
    if name not in sets:
        raise RuntimeError(f"Unknown candidate set {name!r}; available: {', '.join(sorted(sets))}")
    return name, list(manifest.get("common_overlays", [])) + list(sets[name]["overlays"])


def prepare(source, dry_run=False, overlay_set=None):
    source = source.resolve()
    manifest = load_manifest()
    set_name, overlays = resolve_overlays(manifest, overlay_set)
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
    for item in overlays:
        origin, target = REPO / item["source"], source / item["path"]
        if not origin.is_file():
            raise RuntimeError(f"Missing repository source: {origin}")
        expected, actual = digest(origin), digest(target)
        if actual == expected:
            continue
        # A tree prepared with another candidate set of this repository may be switched.
        if actual != item["before_sha256"] and actual not in item.get("accepted_previous_sha256", []):
            raise RuntimeError(f"Unexpected source content: {target}; use a dedicated pinned checkout")
        changes.append(("copy", item["path"], origin, expected))
    print(f"candidate set: {set_name}", flush=True)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mnn-source", type=Path, required=True)
    parser.add_argument("--candidate-set", help="Overlay set name from scripts/patches/manifest.json (default: manifest default)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-sets", action="store_true", help="Print the available overlay sets and exit")
    args = parser.parse_args()
    if args.list_sets:
        manifest = load_manifest()
        for name, item in manifest.get("overlay_sets", {}).items():
            marker = " (default)" if name == manifest.get("default_overlay_set") else ""
            print(f"{name}{marker}: {item.get('description', '')}")
        return
    prepare(args.mnn_source, args.dry_run, args.candidate_set)


if __name__ == "__main__":
    main()
