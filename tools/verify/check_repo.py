#!/usr/bin/env python3
"""Repository convention and consistency checks (docs/CONVENTIONS.md).

Runs on both branch kinds (root workload.json: branch_kind main|workload).
Standard library only. Exit status 1 when any ERROR is found; WARN lines do
not fail unless --strict.

Checks (workload branch): JSON parse + BOM, schema validation of record files,
candidate index <-> experiments/ directories, candidate_id naming, source file
hashes, overlay manifest consistency, run index <-> benchmarks/history/, run_id
naming, frozen-evidence manifest hashes, Markdown links (old paths in frozen
evidence resolved via experiments/path_moves.json), generated-file freshness,
file size limits, Python syntax.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "schemas"
FROZEN = ("benchmarks/history/", "reports/historical/")
CANDIDATE_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9]+(-[a-z0-9]+)*-(l1|l2|l3|profile)$")
TEXT_EXT = {".md", ".json", ".jsonl", ".csv", ".py", ".sh", ".cl", ".cpp", ".hpp", ".cmake", ".yml", ".yaml", ".txt"}
SINGLE_FILE_LIMIT = 8 * 1024 * 1024
RUN_DIR_LIMIT = 32 * 1024 * 1024
COMPRESS_HINT = 1024 * 1024


class Report:
    def __init__(self):
        self.errors, self.warnings, self.oks = [], [], []

    def error(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def ok(self, msg): self.oks.append(msg)


# ---------------------------------------------------------------- utilities
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tracked_files(root=ROOT):
    try:
        out = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True)
        return [p for p in out.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def load_json(path, report):
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        report.error(f"{path}: cannot read ({exc})")
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        rel = os.path.relpath(path, ROOT)
        (report.warn if rel.startswith(FROZEN) else report.error)(f"{rel}: UTF-8 BOM")
        raw = raw[3:]
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        report.error(f"{os.path.relpath(path, ROOT)}: invalid JSON ({exc})")
        return None


TYPES = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool, "null": type(None)}


def validate(instance, schema, path="$"):
    """Minimal JSON-schema (draft-07 subset) validator; returns a list of problems."""
    problems = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        ok = False
        for name in types:
            py = TYPES[name]
            if name == "integer" and isinstance(instance, bool):
                continue
            if name == "number" and isinstance(instance, bool):
                continue
            if isinstance(instance, py):
                ok = True
        if not ok:
            problems.append(f"{path}: expected {t}, got {type(instance).__name__}")
            return problems
    if "enum" in schema and instance not in schema["enum"]:
        problems.append(f"{path}: {instance!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str) and not re.search(schema["pattern"], instance):
        problems.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                problems.append(f"{path}: missing required key {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                problems += validate(instance[key], sub, f"{path}.{key}")
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            problems += validate(item, schema["items"], f"{path}[{i}]")
    return problems


def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


def check_schema(path, name, report):
    data = load_json(path, report)
    if data is None:
        return None
    problems = validate(data, schema(name))
    rel = os.path.relpath(path, ROOT)
    for p in problems:
        report.error(f"{rel}: {p}")
    if not problems:
        report.ok(f"schema {name}: {rel}")
    return data


# ---------------------------------------------------------------- checks
def check_json_and_bom(files, report):
    bad = 0
    for rel in files:
        ext = Path(rel).suffix
        if ext in {".json"}:
            if load_json(ROOT / rel, report) is None:
                bad += 1
        elif ext in {".md", ".csv", ".jsonl", ".py", ".sh", ".yml", ".yaml"} and not rel.startswith(FROZEN):
            with open(ROOT / rel, "rb") as f:
                if f.read(3) == b"\xef\xbb\xbf":
                    report.error(f"{rel}: UTF-8 BOM")
    if not bad:
        report.ok(f"{sum(1 for f in files if f.endswith('.json'))} JSON files parse")


def check_sizes(files, report):
    run_sizes = {}
    for rel in files:
        size = (ROOT / rel).stat().st_size
        if size > SINGLE_FILE_LIMIT:
            report.error(f"{rel}: {size / 2**20:.1f} MiB exceeds the 8 MiB single-file limit (use a Release asset)")
        elif size > COMPRESS_HINT and Path(rel).suffix in {".jsonl", ".csv"} and not rel.startswith(FROZEN):
            report.warn(f"{rel}: {size / 2**20:.1f} MiB {Path(rel).suffix}; compress to .gz before committing")
        if rel.startswith("benchmarks/history/"):
            run = "/".join(rel.split("/")[:3])
            run_sizes[run] = run_sizes.get(run, 0) + size
    for run, size in run_sizes.items():
        if size > RUN_DIR_LIMIT:
            report.error(f"{run}: {size / 2**20:.1f} MiB exceeds the 32 MiB run-directory limit")
    report.ok("file size limits")


def check_python_syntax(files, report):
    pys = [str(ROOT / f) for f in files if f.endswith(".py")]
    if not pys:
        return
    proc = subprocess.run([sys.executable, "-m", "py_compile", *pys], capture_output=True, text=True)
    if proc.returncode:
        report.error("python syntax: " + proc.stderr.strip().splitlines()[-1])
    else:
        report.ok(f"{len(pys)} Python files compile")


def load_path_moves(report):
    path = ROOT / "experiments/path_moves.json"
    if not path.is_file():
        return {}
    data = check_schema(path, "path_moves", report) or {}
    return {m["from"]: m["to"] for m in data.get("moves", [])}


def check_markdown_links(files, moves, report, branch_kind="workload"):
    link_re = re.compile(r"\]\(([^)#\s]+)(#[^)]*)?\)")
    broken, stale = 0, 0
    for rel in files:
        if not rel.endswith(".md") or rel == "README.html":
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for m in link_re.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (ROOT / rel).parent / target
            if resolved.exists():
                continue
            if rel.startswith(FROZEN):
                normalized = os.path.normpath(os.path.relpath(resolved, ROOT))
                if normalized in moves and (ROOT / moves[normalized]).exists():
                    stale += 1
                    continue
                report.warn(f"{rel}: stale link {target} in frozen evidence (not editable)")
                continue
            if branch_kind == "main":
                # Shared docs may point at files that only exist on workload branches.
                report.warn(f"{rel}: link {target} targets a workload-branch file (absent on main)")
                continue
            broken += 1
            report.error(f"{rel}: broken link {target}")
    report.ok(f"markdown links ({stale} old paths in frozen evidence resolved via path_moves.json)")


def check_history_manifests(report):
    root = ROOT / "benchmarks/history"
    if not root.is_dir():
        return
    checked = 0
    for manifest in sorted(root.glob("*/manifest.json")):
        data = load_json(manifest, report)
        if not isinstance(data, dict):
            continue
        entries = []
        for key in ("files", "source_files", "derived_files"):
            if isinstance(data.get(key), list):
                entries += data[key]
        for entry in entries:
            rel = entry.get("path") or entry.get("published_path")
            digest = entry.get("sha256") or entry.get("published_sha256")
            if not rel or not digest:
                continue
            target = manifest.parent / rel
            if not target.is_file():
                target = ROOT / rel
            if not target.is_file():
                report.warn(f"{manifest.parent.name}: listed file missing: {rel}")
                continue
            if sha256(target) != digest:
                report.error(f"{manifest.parent.name}: hash mismatch for frozen evidence {rel}")
            checked += 1
    report.ok(f"frozen evidence hashes: {checked} files verified")


def check_candidates(report):
    index_path = ROOT / "experiments/index.json"
    if not index_path.is_file():
        report.error("experiments/index.json missing")
        return {}
    index = check_schema(index_path, "candidate_index", report) or {"candidates": []}
    indexed = {c["candidate_id"]: c for c in index.get("candidates", [])}
    dirs = {p.parent.name: p for p in (ROOT / "experiments").glob("*/candidate.json")}
    for cid in sorted(set(dirs) - set(indexed)):
        report.error(f"experiments/{cid}/candidate.json is not listed in experiments/index.json")
    for cid in sorted(set(indexed) - set(dirs)):
        report.error(f"experiments/index.json lists {cid} but experiments/{cid}/candidate.json is missing")
    for cid, path in sorted(dirs.items()):
        if not CANDIDATE_ID_RE.match(cid):
            report.error(f"candidate id {cid!r} is not lowercase kebab-case")
        data = check_schema(path, "candidate", report)
        if not data:
            continue
        if data.get("candidate_id") != cid:
            report.error(f"{path.relative_to(ROOT)}: candidate_id {data.get('candidate_id')!r} != directory {cid!r}")
        for key in ("design", "principle"):
            target = data.get(key)
            if target and not (ROOT / target).is_file():
                report.error(f"{cid}: {key} path missing: {target}")
        for entry in data.get("source_files", []):
            src = ROOT / entry["path"]
            if not src.is_file():
                report.error(f"{cid}: source file missing: {entry['path']}")
            elif sha256(src) != entry["sha256"]:
                report.error(f"{cid}: sha256 mismatch for {entry['path']} (update candidate.json or restore the file)")
        if cid in indexed:
            entry = indexed[cid]
            for key in ("status", "candidate_kind", "parent_candidate_id"):
                if entry.get(key) != data.get(key):
                    report.error(f"{cid}: index.{key}={entry.get(key)!r} but candidate.json has {data.get(key)!r}")
            for src_dir in entry.get("sources", []):
                if not (ROOT / src_dir).exists():
                    report.error(f"{cid}: index sources path missing: {src_dir}")
    report.ok(f"candidates: {len(dirs)} directories consistent with experiments/index.json")
    return indexed


def check_overlay_manifest(indexed, report):
    path = ROOT / "scripts/patches/manifest.json"
    if not path.is_file():
        return
    data = load_json(path, report)
    if not isinstance(data, dict):
        return
    if "overlay_sets" not in data:
        report.warn("scripts/patches/manifest.json still uses schema 1 (flat overlays)")
        return
    for item in data.get("patches", []):
        if not (ROOT / "scripts/patches" / item["patch"]).is_file():
            report.error(f"manifest patch missing: {item['patch']}")
    for item in data.get("common_overlays", []):
        if not (ROOT / item["source"]).is_file():
            report.error(f"manifest common overlay missing: {item['source']}")
    sets_used = set()
    for name, item in data["overlay_sets"].items():
        for overlay in item.get("overlays", []):
            if not (ROOT / overlay["source"]).is_file():
                report.error(f"overlay set {name}: source missing: {overlay['source']}")
            if "/overlays/" not in overlay["source"] and not overlay["source"].startswith("kernels/"):
                report.warn(f"overlay set {name}: {overlay['source']} is outside src/<engine>/overlays/<candidate-id>/ or kernels/<candidate-id>/")
        for cid in item.get("candidates", []):
            if cid not in indexed:
                report.error(f"overlay set {name} references unknown candidate {cid}")
            elif name not in indexed[cid].get("overlay_sets", []):
                report.error(f"candidate {cid} does not list overlay set {name} in experiments/index.json")
        sets_used.add(name)
    if data.get("default_overlay_set") not in sets_used:
        report.error("manifest default_overlay_set is not a defined set")
    report.ok(f"overlay manifest: sets {sorted(sets_used)}")


def check_runs(indexed, report):
    index_path = ROOT / "benchmarks/history/index.json"
    if not index_path.is_file():
        report.error("benchmarks/history/index.json missing")
        return
    index = check_schema(index_path, "run_index", report) or {"entries": []}
    entries = {e["run_id"]: e for e in index.get("entries", [])}
    dirs = {p.name for p in (ROOT / "benchmarks/history").iterdir() if p.is_dir()}
    # Legacy entries may name a directory that differs from run_id (frozen, not renamed).
    dir_of = {rid: Path(e.get("directory", f"benchmarks/history/{rid}")).name for rid, e in entries.items()}
    for name in sorted(dirs - set(dir_of.values())):
        report.error(f"benchmarks/history/{name}/ is not listed in benchmarks/history/index.json")
    for rid, name in sorted(dir_of.items()):
        if name not in dirs:
            report.error(f"run index lists {rid} but benchmarks/history/{name}/ is missing")
        if entries[rid].get("directory") and not entries[rid].get("legacy_id"):
            report.error(f"run {rid}: 'directory' is only allowed on legacy_id entries")
    referenced = {r for c in indexed.values() for r in c.get("runs", [])}
    for rid, entry in sorted(entries.items()):
        if not entry.get("legacy_id") and not RUN_ID_RE.match(rid):
            report.error(f"run id {rid!r} does not match YYYYMMDD-HHMMSS-<task>-<level>")
        if not (ROOT / entry["path"]).exists():
            report.error(f"run {rid}: path missing: {entry['path']}")
        run_record = entry.get("run_record")
        if run_record and entry.get("legacy_id"):
            if load_json(ROOT / run_record, report) is None:
                report.error(f"run {rid}: legacy run record unreadable: {run_record}")
        elif run_record:
            check_schema(ROOT / run_record, "run", report)
        elif not entry.get("legacy_id"):
            report.error(f"run {rid}: non-legacy entry without run_record")
        if entry["candidate_id"] not in indexed:
            report.error(f"run {rid}: candidate {entry['candidate_id']} not in experiments/index.json")
        elif rid not in referenced:
            report.warn(f"run {rid}: no candidate lists it in candidate.json/index runs")
    report.ok(f"runs: {len(entries)} index entries consistent with benchmarks/history/")


def check_generated(report):
    embed = ROOT / "tools/embed_kernel.py"
    if embed.is_file():
        proc = subprocess.run([sys.executable, "-I", str(embed), "--check"], capture_output=True, text=True, cwd=ROOT)
        if proc.returncode:
            report.error("embedded kernel headers are stale: run python3 tools/embed_kernel.py")
        else:
            report.ok("embedded kernel headers match their .cl sources")
    html = ROOT / "README.html"
    if html.is_file():
        digest = sha256(ROOT / "README.md")
        if digest not in html.read_text(encoding="utf-8", errors="replace"):
            report.warn("README.html does not reference the current README.md hash; regenerate with scripts/export_readme.py on a Pandoc host")
        else:
            report.ok("README.html matches README.md")


def check_principles(indexed, report):
    root = ROOT / "principles"
    if not root.is_dir():
        return
    for doc in sorted(root.glob("[0-9][0-9]-*.md")):
        cid = doc.stem[3:]
        if cid not in indexed:
            report.warn(f"principles/{doc.name}: no candidate {cid!r} in experiments/index.json")
    report.ok("principles naming")


def check_main_branch(files, report):
    required = ["docs/SOP.md", "docs/CONVENTIONS.md", "docs/SANDBOX.md", "tools/verify/check_repo.py",
                "tools/sandbox/cli.py", "experiments/templates/run.template.json", "schemas/candidate.schema.json"]
    for rel in required:
        if rel not in files:
            report.error(f"main branch is missing shared file {rel}")
    if ".github/workflows/check.yml" not in files:
        report.warn("main branch has no .github/workflows/check.yml (CI workflow); add it once the push token has the workflow scope")
    forbidden = [f for f in files if f.startswith(("kernels/", "src/mnn/", "benchmarks/history/", "reports/", "experiments/")) and not f.startswith("experiments/templates/")]
    for rel in forbidden[:20]:
        report.error(f"main branch must not carry workload content: {rel}")
    if not forbidden:
        report.ok("main branch carries shared tooling only")


# ---------------------------------------------------------------- entry
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument("--mnn-source", type=Path, help="optional pinned MNN checkout: also run scripts/prepare_mnn.py --dry-run for every overlay set")
    args = parser.parse_args()
    report = Report()
    files = tracked_files()
    workload = check_schema(ROOT / "workload.json", "workload", report) if (ROOT / "workload.json").is_file() else None
    if workload is None:
        report.error("workload.json missing at repository root (branch_kind main|workload)")
        kind = "workload"
    else:
        kind = workload["branch_kind"]
    check_json_and_bom(files, report)
    check_sizes(files, report)
    check_python_syntax(files, report)
    moves = load_path_moves(report)
    check_markdown_links(files, moves, report, kind)
    if kind == "workload":
        indexed = check_candidates(report)
        check_overlay_manifest(indexed, report)
        check_runs(indexed, report)
        check_history_manifests(report)
        check_generated(report)
        check_principles(indexed, report)
        if args.mnn_source:
            manifest = load_json(ROOT / "scripts/patches/manifest.json", report) or {}
            for name in manifest.get("overlay_sets", {"default": None}):
                cmd = [sys.executable, str(ROOT / "scripts/prepare_mnn.py"), "--mnn-source", str(args.mnn_source), "--dry-run"]
                if name != "default":
                    cmd += ["--candidate-set", name]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                (report.ok if proc.returncode == 0 else report.error)(f"prepare_mnn dry-run [{name}]: {proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-200:]}")
    else:
        check_main_branch(files, report)
    for line in report.oks:
        print("OK   ", line)
    for line in report.warnings:
        print("WARN ", line)
    for line in report.errors:
        print("ERROR", line)
    failed = bool(report.errors) or (args.strict and report.warnings)
    print(f"\n{len(report.oks)} ok, {len(report.warnings)} warnings, {len(report.errors)} errors -> {'FAIL' if failed else 'PASS'} (branch kind: {kind})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
