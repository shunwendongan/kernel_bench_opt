#!/usr/bin/env python3
"""Summarize retrieved raw ConvTranspose A/B logs; never read old summaries.

Input: explicit --bench-root with T*_profile*/results.jsonl.
Optional --e2e accepts the directory containing results.jsonl, run.json and
comparisons.json produced by omni_ab. Output is descriptive, with no automatic
significance or stable-tail-latency claims. All numerical times retain the raw
unit (microseconds for the operator, seconds for full requests).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def percentile_linear(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def stats(values: list[float], *, p95: bool = True) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "p95": None,
                "min": None, "max": None, "std": None}
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values),
            "p95": percentile_linear(values, .95) if p95 else None,
            "min": min(values), "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else None}


def finite_number(value: Any, *, positive: bool = False) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value) and (value > 0 if positive else value >= 0)


def paired_delta(old: float, new: float) -> dict[str, float | None]:
    return {"old": old, "new": new, "saved": old - new,
            "speedup": old / new if new > 0 else None,
            "reduction_pct": 100 * (1 - new / old) if old > 0 else None}


def paired_summary(pairs: list[dict[str, Any]], *, p95: bool = True) -> dict[str, Any]:
    old = stats([p["old"] for p in pairs], p95=p95)
    new = stats([p["new"] for p in pairs], p95=p95)
    out = {"old": old, "new": new}
    for key in ("saved", "speedup", "reduction_pct"):
        out["paired_" + key] = stats([p[key] for p in pairs if p[key] is not None], p95=p95)
    out["ratio_of_medians"] = old["median"] / new["median"] if new["median"] else None
    out["reduction_of_medians_pct"] = 100 * (1 - new["median"] / old["median"]) if old["median"] else None
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_micro(bench_root: Path) -> dict[str, Any]:
    files = sorted(p for p in bench_root.glob("T*_profile*/results.jsonl")
                   if re.fullmatch(r"T\d+_profile[01]", p.parent.name))
    if not files:
        raise FileNotFoundError(f"No formal raw benchmark JSONL under {bench_root}")
    cases = []
    for path in files:
        rows = load_jsonl(path)
        errors = []
        manifests = [r for r in rows if r.get("kind") == "manifest"]
        if len(manifests) != 1:
            raise ValueError(f"{path}: expected exactly one manifest")
        manifest = manifests[0]
        expected_t, expected_profile = map(int, re.fullmatch(r"T(\d+)_profile([01])", path.parent.name).groups())
        if manifest.get("T") != expected_t or manifest.get("profiling_queue") is not bool(expected_profile):
            errors.append("Manifest T/profiling_queue disagrees with case directory")
        if manifest.get("warmup") != 3 or manifest.get("reps") != 15 or manifest.get("tile") != 4:
            errors.append("Manifest does not match the formal 3-warmup/15-pair/tile4 protocol")
        if manifest.get("axis") != "h" or manifest.get("old_lws") != [0, 0, 0] or manifest.get("new_lws") != [8, 4, 1]:
            errors.append("Manifest axis/LWS differ from the captured production baseline and formal candidate")
        if manifest.get("old_gws") != [192, 1, 5 * expected_t] or manifest.get("new_gws") != [192, (expected_t + 3) // 4, 5]:
            errors.append("Manifest global work sizes differ from the formal shape contract")
        correctness = [r for r in rows if r.get("kind") == "correctness"]
        capture_checks = [r for r in rows if r.get("kind") == "baseline_vs_mnn_capture"]
        correctness_ok = len(correctness) == 1 and all(
            r.get("passed") is True and r.get("bit_mismatches") == 0
            and r.get("numeric_failures") == 0 and r.get("nonfinite") == 0
            and r.get("max_abs") == 0 and r.get("rmse") == 0
            for r in correctness)
        capture_ok = len(capture_checks) == 1 and all(
            r.get("passed") is True and r.get("numeric_mismatches") == 0
            and r.get("max_abs") == 0 for r in capture_checks)
        if not correctness_ok:
            errors.append("Missing/failed bitwise candidate correctness record")
        if not capture_ok:
            errors.append("Missing/failed baseline-versus-MNN-capture record")
        by_rep: dict[int, dict[str, Any]] = {}
        samples = [r for r in rows if r.get("kind") == "sample"]
        for row in samples:
            rep, backend = row.get("rep"), row.get("backend")
            if not isinstance(rep, int) or rep < 0 or backend not in ("old", "candidate"):
                errors.append(f"Invalid sample identity: rep={rep}, backend={backend}")
                continue
            if backend in by_rep.setdefault(rep, {}):
                errors.append(f"Duplicate sample rep={rep}, backend={backend}")
            by_rep[rep][backend] = row
            if row.get("T") != expected_t:
                errors.append(f"Sample T mismatch at rep={rep}")
            if not finite_number(row.get("host_us"), positive=True):
                errors.append(f"Invalid host_us at rep={rep}, backend={backend}")
            if expected_profile and not finite_number(row.get("kernel_us"), positive=True):
                errors.append(f"Missing/invalid profiled kernel_us at rep={rep}, backend={backend}")
            if not expected_profile and row.get("kernel_us") is not None:
                errors.append(f"Unexpected kernel_us on clean queue at rep={rep}, backend={backend}")
        expected_reps = manifest.get("reps")
        if not isinstance(expected_reps, int) or set(by_rep) != set(range(expected_reps)):
            errors.append("Repetition IDs/count do not match manifest")
        pairs = []
        for rep, pair in sorted(by_rep.items()):
            if set(pair) != {"old", "candidate"}:
                errors.append(f"Incomplete pair {rep}")
                continue
            old, new = pair["old"], pair["candidate"]
            if {old.get("order"), new.get("order")} != {0, 1}:
                errors.append(f"Invalid execution order for pair {rep}")
            if old.get("order") != rep % 2:
                errors.append(f"Unexpected alternating order for pair {rep}")
            for metric, domain in (("host_us", "host_enqueue_to_finish"), ("kernel_us", "device_event")):
                if not finite_number(old.get(metric), positive=True) or not finite_number(new.get(metric), positive=True):
                    continue
                pairs.append({"rep": rep, "domain": domain, "old_order": old["order"],
                              "new_order": new["order"], **paired_delta(old[metric], new[metric])})
        summaries = {domain: paired_summary([p for p in pairs if p["domain"] == domain])
                     for domain in ("host_enqueue_to_finish", "device_event")
                     if any(p["domain"] == domain for p in pairs)}
        # Failed correctness/protocol logs remain auditable, but cannot supply
        # apparent speedups for a downstream report that misses the audit flag.
        if errors:
            summaries, pairs = {}, []
        cases.append({"case": path.parent.name, "source": source(path), "manifest": manifest,
                      "correctness_records": correctness, "baseline_vs_mnn_capture": capture_checks,
                      "sample_rows": len(samples), "paired_rows": len(by_rep),
                      "audit_passed": not errors, "gain_eligible": not errors, "audit_errors": errors,
                      "summaries_us": summaries, "pairs_us": pairs})
    expected_cases = {(52, 0), (52, 1), (120, 0), (120, 1), (136, 0), (136, 1)}
    seen = {(c["manifest"]["T"], int(c["manifest"]["profiling_queue"])) for c in cases}
    coverage_ok = seen == expected_cases
    return {"schema_version": 1, "source_policy": "Retrieved raw results.jsonl only; previous summaries not read",
            "timing_units": "microseconds", "std_definition": "sample standard deviation, ddof=1",
            "p95_definition": "linear interpolation at (n-1)*0.95; descriptive statistic of 15 measurements",
            "primary_result": "profile0 host_enqueue_to_finish; profiled device_event is explanatory only",
            "timing_domain_rule": "Host and device timings are separate overlapping domains and must not be added",
            "speedup_definition": "old/new; reduction_pct=100*(1-new/old); saved=old-new",
            "expected_cases": sorted([list(x) for x in expected_cases]), "coverage_passed": coverage_ok,
            "audit_passed": coverage_ok and all(c["audit_passed"] for c in cases),
            "cases": cases}


E2E_METRICS = (
    "request_s", "ttfa_s", "token2wav_s", "thinker_response_s", "waveform_phase_s",
    "audio_encoder_including_prepare_s", "thinker_prefill_s", "thinker_decode_s",
    "thinker_sample_s", "talker_prefill_s", "talker_decode_s", "talker_sample_s",
)


def summarize_e2e(root: Path) -> dict[str, Any]:
    result_path, state_path, check_path = [root / n for n in ("results.jsonl", "run.json", "comparisons.json")]
    rows, state, comparisons = load_jsonl(result_path), json.loads(state_path.read_text()), json.loads(check_path.read_text())
    errors = []
    evidence_sources = [source(p) for p in (result_path, state_path, check_path)]
    configs = {}
    for variant in ("old", "new"):
        path = root / f"effective_config_{variant}.json"
        if not path.is_file():
            errors.append(f"Missing effective config for {variant}")
            continue
        configs[variant] = json.loads(path.read_text())
        evidence_sources.append(source(path))
    config_differences = []
    if set(configs) == {"old", "new"}:
        old_config, new_config = configs["old"], configs["new"]
        for key in sorted(old_config.keys() | new_config.keys()):
            if old_config.get(key) != new_config.get(key):
                config_differences.append({"key": key, "old": old_config.get(key),
                                           "new": new_config.get(key), "allowed": key == "tmp_path"})
        if any(not d["allowed"] for d in config_differences):
            errors.append("Effective inference configs differ beyond per-instance tmp_path")
        if not old_config.get("tmp_path") or not new_config.get("tmp_path") or old_config["tmp_path"] == new_config["tmp_path"]:
            errors.append("Expected separate nonempty per-instance tmp_path values")
    if state.get("distinct_executor_objects") is not True or state.get("resident_instances") != 2:
        errors.append("run.json does not confirm two distinct resident executor objects")
    if state.get("warmup_per_instance") != 1 or state.get("tile") != 4:
        errors.append("run.json warmup/tile protocol differs from the planned test")
    if state.get("profile_enabled") is not False or state.get("capture_enabled") is not False:
        errors.append("run.json does not confirm clean timing without profiling/capture")
    if state.get("seed") is None or not str(state["seed"]).strip():
        errors.append("Missing reproducibility seed in run.json")
    dispatch = {}
    guarded_shapes = {52, 120, 136}
    for variant in ("old", "new"):
        path = root / f"warmup_{variant}/dispatch.jsonl"
        if not path.is_file():
            errors.append(f"Missing warmup actual dispatch evidence for {variant}")
            continue
        dispatch[variant] = load_jsonl(path)
        evidence_sources.append(source(path))
        if not dispatch[variant]:
            errors.append(f"Empty warmup dispatch evidence for {variant}")
        for d in dispatch[variant]:
            expected_candidate = variant == "new" and d.get("T") in guarded_shapes
            if d.get("candidate") is not expected_candidate or d.get("precision") != 0 or d.get("tile") != (4 if variant == "new" else 0):
                errors.append(f"Incorrect warmup dispatch/precision for {variant}, T={d.get('T')}")
    if set(dispatch) == {"old", "new"}:
        if [d.get("T") for d in dispatch["old"]] != [d.get("T") for d in dispatch["new"]]:
            errors.append("Warmup old/new dispatch shape sequences differ")
        if not any(d.get("candidate") is True for d in dispatch["new"]):
            errors.append("No actual candidate dispatch hit in new warmup")
    if not isinstance(comparisons, list):
        raise ValueError(f"{check_path}: expected array")
    warmup = [r for r in rows if r.get("warmup") is True]
    measured = [r for r in rows if r.get("warmup") is False and isinstance(r.get("pair"), int) and r["pair"] >= 0]
    if len(warmup) != 2 or {r.get("variant") for r in warmup} != {"old", "new"}:
        errors.append("Expected one warmup request per variant")
    if len(warmup) + len(measured) != len(rows):
        errors.append("Rows with unknown warmup/pair status")
    expected_order = [(-1, "old"), (-1, "new")]
    for pair in range(3):
        expected_order.extend((pair, variant) for variant in (("new", "old") if pair % 2 else ("old", "new")))
    observed_order = [(r.get("pair"), r.get("variant")) for r in rows]
    if observed_order != expected_order:
        errors.append("Actual request order/count differs from two warmups plus three alternating pairs")
    by_pair: dict[int, dict[str, Any]] = {}
    for r in measured:
        pair, variant = r["pair"], r.get("variant")
        if variant not in ("old", "new"):
            errors.append(f"Unknown E2E variant {variant}")
            continue
        if variant in by_pair.setdefault(pair, {}):
            errors.append(f"Duplicate E2E pair={pair}, variant={variant}")
        by_pair[pair][variant] = r
    expected = state.get("pairs")
    if expected != 3:
        errors.append(f"Expected planned 3 measured pairs, run.json has {expected}")
    if set(by_pair) != {0, 1, 2}:
        errors.append("Missing/extra measured pair IDs")
    if state.get("state") != "completed" or state.get("measured_pairs_completed") != 3:
        errors.append("run.json does not confirm three completed pairs")
    if state.get("all_equality_checks_passed") is not True:
        errors.append("run.json equality checks not passed")
    for r in rows:
        if r.get("success") is not True or r.get("profile_enabled") is not False or r.get("capture_enabled") is not False:
            errors.append(f"Request failed or enabled instrumentation: pair={r.get('pair')}, variant={r.get('variant')}")
        expected_tile = 0 if r.get("variant") == "old" else 4
        if r.get("tile") != expected_tile or r.get("resident_instances") != 2 or r.get("row") != state.get("row"):
            errors.append(f"Request variant/tile/residency/sample differs from run.json: pair={r.get('pair')}, variant={r.get('variant')}")
        if r.get("files_ok") is not True or r.get("wave_last_received") is not True or r.get("nonfinite_samples") != 0 or not finite_number(r.get("wave_samples"), positive=True):
            errors.append(f"Incomplete/nonfinite/missing waveform: pair={r.get('pair')}, variant={r.get('variant')}")
        if r.get("warmup") is not (r.get("pair") == -1):
            errors.append(f"Inconsistent request warmup flag: pair={r.get('pair')}")
        # The three timestamps used by the runner form two adjacent wall-clock
        # intervals. Runtime counters may overlap and are not summed here.
        wall = [r.get(k) for k in ("request_s", "thinker_response_s", "waveform_phase_s", "ttfa_s")]
        if not all(finite_number(x) for x in wall) or wall[0] <= 0 or wall[3] > wall[0] or abs(wall[0] - wall[1] - wall[2]) > 1e-5:
            errors.append(f"Inconsistent request/TTFA wall-clock scope: pair={r.get('pair')}, variant={r.get('variant')}")
    scopes = {r.get("request_scope") for r in rows}
    if len(scopes) != 1 or not all(isinstance(s, str) and s for s in scopes):
        errors.append("Request timing scope missing or differs across variants/repetitions")
    if len({r.get("id") for r in rows}) != 1 or any(r.get("id") is None for r in rows):
        errors.append("Sample identity differs across warmup/measured requests")
    check_records = []
    seen_comparisons = []
    for row in comparisons:
        pair, is_warmup = row.get("pair"), row.get("warmup")
        seen_comparisons.append(pair)
        if is_warmup is not (pair == -1) or row.get("new_first") is not (isinstance(pair, int) and pair >= 0 and pair % 2 == 1):
            errors.append(f"Comparison warmup/order metadata inconsistent for pair={pair}")
        keys = ("paired",) if is_warmup is True else ("paired", "old_vs_warmup_workload", "new_vs_warmup_workload")
        for key in keys:
            check = row.get(key, {})
            passed = all(check.get(k) is True for k in ("passed", "text_tokens_equal", "speech_tokens_equal", "wave_size_equal", "wave_bitwise_equal"))
            passed = passed and all(check.get(k) == 0 for k in ("wave_max_abs", "wave_rmse", "wave_relative_rmse"))
            record = {"pair": pair, "warmup": is_warmup, "comparison": key, "verified_passed": passed, "raw": check}
            check_records.append(record)
            if not passed:
                errors.append(f"Missing/failed correctness check pair={pair}, comparison={key}")
    if not all(isinstance(p, int) for p in seen_comparisons) or sorted(seen_comparisons) != [-1, 0, 1, 2]:
        errors.append("Missing/duplicate/extra E2E comparisons")
    pairs = []
    for pair_id, pair in sorted(by_pair.items()):
        if set(pair) != {"old", "new"}:
            errors.append(f"Incomplete E2E pair {pair_id}")
            continue
        old, new = pair["old"], pair["new"]
        for identity in ("id", "row", "prompt_tokens", "text_token_ids", "speech_token_ids", "wave_samples", "wave_chunks"):
            if identity not in old or identity not in new or old[identity] != new[identity]:
                errors.append(f"Raw request workload mismatch or missing field pair={pair_id}, field={identity}")
        for metric in E2E_METRICS:
            if not finite_number(old.get(metric)) or not finite_number(new.get(metric)):
                errors.append(f"Missing/nonfinite E2E metric {metric}, pair={pair_id}")
                continue
            pairs.append({"pair": pair_id, "metric": metric, **paired_delta(old[metric], new[metric])})
    summaries = {metric: paired_summary([p for p in pairs if p["metric"] == metric], p95=False)
                 for metric in E2E_METRICS if any(p["metric"] == metric for p in pairs)}
    if errors:
        pairs, summaries = [], {}
    request_fields = ("variant", "pair", "warmup", "row", "id", "success", "profile_enabled", "capture_enabled",
                      "prompt_tokens", "text_tokens", "speech_tokens", "wave_samples", "wave_chunks", "wave_duration_s",
                      "thinker_truncated", "thinker_status", "talker_status", "nonfinite_samples", "files_ok",
                      "resident_instances", "process_rss_before_kib", "process_rss_after_kib", "memavailable_before_kib",
                      "memavailable_after_kib", "process_max_rss_kib", "request_scope", "dit_timing_note")
    return {"schema_version": 1, "sources": evidence_sources,
            "timing_units": "seconds", "std_definition": "sample standard deviation, ddof=1",
            "tail_statistic_policy": "No E2E p95 reported: n=3 pairs is too small for stable tail-latency estimation",
            "inference_policy": "Descriptive paired changes only; no significance claim or confidence claim",
            "counter_policy": "Request/TTFA are wall-time metrics; native runtime stage counters are not added as an E2E decomposition",
            "warmup_excluded_from_timing": True, "warmup_requests": len(warmup), "measured_requests": len(measured),
            "run": state, "audit_passed": not errors, "gain_eligible": not errors, "audit_errors": errors,
            "fairness": {"config_differences": config_differences, "allowed_config_difference": "tmp_path only",
                         "observed_order": observed_order, "expected_order": expected_order,
                         "request_scopes": sorted(scopes, key=str), "dispatch": dispatch,
                         "dispatch_scope": "Warmup onEncode evidence; not a count of every timed invocation",
                         "guarded_shapes": sorted(guarded_shapes),
                         "isolation_evidence_limit": "Distinct executor pointer flag does not prove global runtime-state isolation"},
            "correctness": check_records,
            "request_statuses": [{k: r.get(k) for k in request_fields} for r in rows],
            "pairs_s": pairs, "summaries_s": summaries}


def micro_csv(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats_rows, pair_rows = [], []
    for case in summary["cases"]:
        common = {"case": case["case"], "T": case["manifest"]["T"],
                  "profiling_queue": case["manifest"]["profiling_queue"], "audit_passed": case["audit_passed"]}
        for domain, sums in case["summaries_us"].items():
            for backend in ("old", "new"):
                stats_rows.append({**common, "domain": domain, "backend": backend, "unit": "us", **sums[backend],
                                   "ratio_of_medians": sums["ratio_of_medians"],
                                   "reduction_of_medians_pct": sums["reduction_of_medians_pct"],
                                   "median_paired_speedup": sums["paired_speedup"]["median"],
                                   "median_paired_reduction_pct": sums["paired_reduction_pct"]["median"]})
        pair_rows.extend({**common, "unit": "us", **p} for p in case["pairs_us"])
    return stats_rows, pair_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--e2e", type=Path, help="Directory with raw results.jsonl, run.json and comparisons.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    micro = summarize_micro(args.bench_root)
    write_json(args.out / "microbenchmark_summary.json", micro)
    stat_rows, pair_rows = micro_csv(micro)
    write_csv(args.out / "microbenchmark_stats.csv", stat_rows)
    write_csv(args.out / "microbenchmark_pairs.csv", pair_rows)
    status = {"microbenchmark_audit_passed": micro["audit_passed"], "formal_cases": len(micro["cases"]),
              "output": str(args.out.resolve())}
    if args.e2e:
        e2e = summarize_e2e(args.e2e)
        write_json(args.out / "e2e_summary.json", e2e)
        write_csv(args.out / "e2e_pairs.csv", e2e["pairs_s"])
        rows = []
        for metric, sums in e2e["summaries_s"].items():
            for backend in ("old", "new"):
                rows.append({"metric": metric, "backend": backend, "unit": "s", **sums[backend],
                             "ratio_of_medians": sums["ratio_of_medians"],
                             "reduction_of_medians_pct": sums["reduction_of_medians_pct"],
                             "median_paired_speedup": sums["paired_speedup"]["median"],
                             "median_paired_reduction_pct": sums["paired_reduction_pct"]["median"],
                             "audit_passed": e2e["audit_passed"]})
        write_csv(args.out / "e2e_stats.csv", rows)
        status["e2e_audit_passed"] = e2e["audit_passed"]
        status["e2e_measured_requests"] = e2e["measured_requests"]
    print(json.dumps(status, ensure_ascii=False))
    return 0 if all(v for k, v in status.items() if k.endswith("audit_passed")) else 3


if __name__ == "__main__":
    raise SystemExit(main())
