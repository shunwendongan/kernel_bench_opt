#!/usr/bin/env python3
"""Rank MNN hotspots from OmniProfile callback JSONL without the external toolkit.

Input: one or more `OMNI_PROFILE_PATH` files (optionally .gz) written by
src/mnn/omni_profile.hpp (events kind=op | module_api | runtime), or an existing
rankings CSV (--from-rankings-csv, read with utf-8-sig). No model is loaded.

Outputs (UTF-8, no BOM):
  rankings.csv      scope, backend, op_type, name, count, total_us, max_us  (same columns as the historical table)
  by_type.csv       per stage x op_type totals
  modules.csv       module_api inclusive totals per module
  top.md            human-readable tables

Durations are synchronized host callback times and include instrumentation
overhead; module_api and op durations overlap and are never added together.
"""
import argparse
import collections
import csv
import gzip
import json
import sys
from pathlib import Path


def read_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, {"kind": "invalid", "raw": line}


def rank(event_iter):
    """Aggregate events; returns dict with rankings/by_type/modules/counts/warnings."""
    ranks, by_type, modules = {}, collections.Counter(), {}
    counts, warnings = collections.Counter(), []
    for line_no, raw in event_iter:
        kind = raw.get("kind", "unknown")
        counts[kind] += 1
        if kind == "runtime":
            continue
        if kind not in ("op", "module_api"):
            warnings.append({"line": line_no, "reason": f"ignored event kind {kind}"})
            continue
        try:
            duration = float(raw["elapsed_us"])
        except (KeyError, TypeError, ValueError):
            warnings.append({"line": line_no, "reason": "missing elapsed_us"})
            continue
        stage = str(raw.get("stage", raw.get("module", "unscoped")))
        executor = str(raw.get("executor", "module_api"))
        backend = "tensor_owners=" + json.dumps(raw.get("output_tensor_backends", []), separators=(",", ":"))
        name = str(raw.get("name", raw.get("module", "[unnamed]"))) or "[unnamed]"
        op_type = str(raw.get("type", "ModuleAPI")) or "[unknown]"
        scope = f"{kind}__{executor}__{stage}"
        key = (scope, backend, op_type, name)
        r = ranks.setdefault(key, {"scope": scope, "backend": backend, "op_type": op_type, "name": name,
                                   "count": 0, "total_us": 0.0, "max_us": 0.0})
        r["count"] += 1
        r["total_us"] += duration
        r["max_us"] = max(r["max_us"], duration)
        if kind == "op":
            by_type[(stage, op_type)] += duration
        else:
            m = modules.setdefault(name, {"module": name, "count": 0, "total_us": 0.0, "max_us": 0.0})
            m["count"] += 1
            m["total_us"] += duration
            m["max_us"] = max(m["max_us"], duration)
    return {
        "rankings": sorted(ranks.values(), key=lambda r: (r["scope"], -r["total_us"])),
        "by_type": sorted(({"stage": s, "op_type": t, "total_us": v} for (s, t), v in by_type.items()),
                          key=lambda r: (r["stage"], -r["total_us"])),
        "modules": sorted(modules.values(), key=lambda m: -m["total_us"]),
        "counts": dict(counts), "warnings": warnings,
    }


def rank_from_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r["count"] = int(r["count"]); r["total_us"] = float(r["total_us"]); r["max_us"] = float(r["max_us"])
            rows.append(r)
    by_type = collections.Counter()
    modules = []
    for r in rows:
        if r["scope"].startswith("op__"):
            by_type[(r["scope"].split("__", 2)[-1], r["op_type"])] += r["total_us"]
        elif r["scope"].startswith("module_api__"):
            modules.append({"module": r["name"], "count": r["count"], "total_us": r["total_us"], "max_us": r["max_us"]})
    return {"rankings": sorted(rows, key=lambda r: (r["scope"], -r["total_us"])),
            "by_type": sorted(({"stage": s, "op_type": t, "total_us": v} for (s, t), v in by_type.items()),
                              key=lambda r: (r["stage"], -r["total_us"])),
            "modules": sorted(modules, key=lambda m: -m["total_us"]), "counts": {"csv_rows": len(rows)}, "warnings": []}


def write_outputs(result, out, top):
    out.mkdir(parents=True, exist_ok=True)
    def dump(name, rows, fields):
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    dump("rankings.csv", result["rankings"], ["scope", "backend", "op_type", "name", "count", "total_us", "max_us"])
    dump("by_type.csv", result["by_type"], ["stage", "op_type", "total_us"])
    dump("modules.csv", result["modules"], ["module", "count", "total_us", "max_us"])
    lines = ["# Hotspot ranking", "",
             "Host synchronized callback durations incl. instrumentation; module_api and op rows overlap and must not be summed.", ""]
    lines += ["## Modules (module_api, inclusive)", "", "| module | count | total s | max s |", "|---|---:|---:|---:|"]
    lines += [f"| {m['module']} | {m['count']} | {m['total_us'] / 1e6:.3f} | {m['max_us'] / 1e6:.3f} |" for m in result["modules"]]
    lines += ["", "## Op types per stage", "", "| stage | op_type | total s |", "|---|---|---:|"]
    lines += [f"| {r['stage']} | {r['op_type']} | {r['total_us'] / 1e6:.3f} |" for r in result["by_type"][:40]]
    ops = sorted((r for r in result["rankings"] if r["scope"].startswith("op__")), key=lambda r: -r["total_us"])[:top]
    lines += ["", f"## Top {top} operators", "", "| stage | op_type | name | count | total s | max ms |", "|---|---|---|---:|---:|---:|"]
    lines += [f"| {r['scope'].split('__', 2)[-1]} | {r['op_type']} | {r['name']} | {r['count']} | {r['total_us'] / 1e6:.3f} | {r['max_us'] / 1e3:.1f} |" for r in ops]
    (out / "top.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"event_counts": result["counts"], "warnings": result["warnings"][:100],
                                                   "rows": len(result["rankings"])}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profiles", nargs="*", type=Path, help="OmniProfile JSONL files (.jsonl or .jsonl.gz)")
    p.add_argument("--from-rankings-csv", type=Path, help="summarize an existing rankings CSV instead of raw events")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()
    if args.from_rankings_csv:
        result = rank_from_csv(args.from_rankings_csv)
    elif args.profiles:
        def events():
            for path in args.profiles:
                yield from read_events(path)
        result = rank(events())
    else:
        p.error("give profile files or --from-rankings-csv")
    write_outputs(result, args.out, args.top)
    print(json.dumps({"rows": len(result["rankings"]), "modules": len(result["modules"]),
                      "events": result["counts"], "warnings": len(result["warnings"]), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
