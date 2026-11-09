#!/usr/bin/env python3
"""Offline analysis of retrieved Omni system-route A/B records.

The historical board runner produced the request records; this portable script
does not build that runner, contact a device, or execute inference. By default it
re-reads raw waveform bytes. The explicit recorded-evidence mode verifies bound
text records and reuses historical waveform checks without re-reading waveforms.
"""
from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import stat
import statistics
import sys


def load_json(path, errors, required=False):
    if not path.is_file():
        if required:
            errors.append(f"Missing record: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read {path}: {exc}")
        return None


def lines(path, errors):
    if not path.is_file():
        errors.append(f"Missing JSONL: {path}")
        return []
    values = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected an object")
            values.append(value)
        except ValueError as exc:
            errors.append(f"{path.name}:{number}: {exc}")
    return values


def audit_runtime_log(path, rows, errors):
    """Bind encoder diagnostics to the ordered request markers; preserve raw files."""
    if not path.is_file():
        errors.append("Missing process.txt/process.log; encoder failures cannot be audited")
        return []
    starts = re.compile(r"^AB_REQUEST_START variant=(\w+) pair=(-?\d+) warmup=([01])$")
    done = re.compile(r"^AB_REQUEST_DONE variant=(\w+) pair=(-?\d+) warmup=([01]) ")
    messages = ("Broad cast error", "Compute Shape Error", "Media encoder returned no features", "RKNN audio prefix inference failed")
    blocks, current = [], None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = starts.match(line)
        if match:
            if current is not None:
                errors.append("Nested request markers in runtime process log")
            current = {"variant": match[1], "pair": int(match[2]), "warmup": bool(int(match[3])), "diagnostics": []}
            blocks.append(current)
        elif current is not None and any(text in line for text in messages):
            current["diagnostics"].append({"source_line": line_number, "message": line})
        elif done.match(line):
            current = None
    if len(blocks) != len(rows):
        errors.append("Request markers and result rows differ; runtime audit is incomplete")
    affected = []
    for block, metric in zip(blocks, rows):
        if any(block[key] != metric.get(key) for key in ["variant", "pair", "warmup"]):
            errors.append("Runtime marker does not match result-row order")
            continue
        metric["runner_reported_success"] = metric.get("success")
        metric["runtime_diagnostics"] = block["diagnostics"]
        metric["valid_audio_features"] = not block["diagnostics"]
        if block["diagnostics"]:
            metric["success"] = False
            affected.append({"row": metric.get("row"), "id": metric.get("id"), "variant": metric.get("variant"), "warmup": metric.get("warmup"), "runner_reported_success": metric["runner_reported_success"], "valid_for_comparison": False, "diagnostics": block["diagnostics"]})
    return affected


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def delta(old, new):
    if not number(old) or not number(new) or old <= 0 or new <= 0:
        return {"old_s": old, "new_s": new, "saved_s": None, "old_over_new": None, "reduction_pct": None}
    return {"old_s": old, "new_s": new, "saved_s": old - new,
            "old_over_new": old / new, "reduction_pct": (1 - new / old) * 100}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bound_file(root, relative):
    """Only accept a regular file below root, without symlink path components."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid binding path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Unsafe binding path: {relative!r}")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Binding traverses a symlink: {relative}")
    try:
        cursor.resolve().relative_to(root.resolve())
        if not stat.S_ISREG(cursor.stat().st_mode):
            raise ValueError(f"Binding is not a regular file: {relative}")
    except OSError as exc:
        raise ValueError(f"Cannot read bound file {relative}: {exc}") from exc
    return cursor


def recorded_waveform_checks(root, path, expected, runtime_path, manifest):
    """Validate sidecar provenance and pair identities before consuming checks."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read recorded waveform checks: {exc}") from exc
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload.get("evidence_kind") != "recorded_device_waveform_validation"):
        raise ValueError("Unsupported recorded waveform checks schema/evidence kind")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Recorded waveform checks require a bindings object")
    if manifest is None:
        raise ValueError("Recorded waveform checks require the frozen input manifest")
    required = {"e2e/results.jsonl", str(runtime_path.relative_to(root)), str(manifest.relative_to(root))}
    if not required.issubset(bindings):
        raise ValueError(f"Missing required waveform evidence bindings: {sorted(required - bindings.keys())}")
    for relative, expected_sha in bindings.items():
        if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            raise ValueError(f"Invalid SHA256 binding for {relative!r}")
        if sha(bound_file(root, relative)) != expected_sha:
            raise ValueError(f"Recorded waveform evidence SHA256 mismatch: {relative}")
    values = payload.get("checks")
    if not isinstance(values, list):
        raise ValueError("Recorded waveform checks must contain a checks array")
    indexed = {}
    for value in values:
        if not isinstance(value, dict) or type(value.get("row")) is not int or type(value.get("pair")) is not int:
            raise ValueError("Each recorded waveform check needs integer row and pair fields")
        key = (value["row"], value["pair"])
        if key in indexed:
            raise ValueError(f"Duplicate recorded waveform check: {key}")
        if key not in expected or value.get("id") != expected[key]:
            raise ValueError(f"Recorded waveform check identity differs from manifest: {key}")
        if not isinstance(value.get("comparison"), dict):
            raise ValueError(f"Missing recorded waveform comparison: {key}")
        indexed[key] = value["comparison"]
    if indexed.keys() != expected.keys():
        raise ValueError(f"Missing recorded waveform checks: {sorted(expected.keys() - indexed.keys())}")
    try:
        display_path = str(path.relative_to(root))
    except ValueError:
        display_path = path.name
    return indexed, {"path": display_path, "bytes": path.stat().st_size, "sha256": sha(path),
                     "evidence_kind": payload["evidence_kind"], "verified_bindings": bindings,
                     "waveform_bytes_rehashed": False}


def recorded_waveform_identity(value, metric, label):
    if not isinstance(value, dict) or metric is None:
        raise ValueError(f"Missing {label} waveform identity/metric in recorded evidence")
    expected_path = str(Path(metric.get("output_directory", "")) / "waveform.f32")
    if value.get("path") != expected_path:
        raise ValueError(f"Recorded {label} waveform path differs from request metric")
    for field in ("bytes", "sample_count", "nonfinite"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError(f"Invalid recorded {label} waveform {field}")
    if value["bytes"] != value["sample_count"] * 4 or value.get("valid_float32_size") is not True:
        raise ValueError(f"Invalid recorded {label} float32 waveform byte count")
    if value["nonfinite"] > value["sample_count"] or value.get("applicable_to_speech") is not True:
        raise ValueError(f"Invalid recorded {label} speech waveform identity")
    if not isinstance(value.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None:
        raise ValueError(f"Invalid recorded {label} waveform SHA256")
    metric_samples = metric.get("wave_samples")
    if type(metric_samples) is not int or metric_samples < 0:
        raise ValueError(f"Invalid {label} wave_samples metric")
    matches = value["sample_count"] == metric_samples
    if type(value.get("matches_metric_samples")) is not bool or value["matches_metric_samples"] != matches:
        raise ValueError(f"Recorded {label} waveform sample binding disagrees with metrics")
    if "nonfinite_samples" in metric and metric["nonfinite_samples"] != value["nonfinite"]:
        raise ValueError(f"Recorded {label} nonfinite count disagrees with metrics")
    fields = ("path", "bytes", "sha256", "applicable_to_speech", "valid_float32_size",
              "sample_count", "nonfinite", "matches_metric_samples")
    return {field: value[field] for field in fields}


def recorded_waveform_comparison(recorded, a, b):
    """Consume only waveform fields; token and success fields remain untrusted."""
    old = recorded_waveform_identity(recorded.get("old_waveform"), a, "old")
    new = recorded_waveform_identity(recorded.get("new_waveform"), b, "new")
    same_size = old["bytes"] == new["bytes"]
    same_bits = same_size and old["sha256"] == new["sha256"]
    for field, computed in (("wave_size_equal", same_size), ("wave_bitwise_equal", same_bits)):
        if type(recorded.get(field)) is not bool or recorded[field] != computed:
            raise ValueError(f"Recorded waveform {field} contradicts recorded identities")
    errors = [recorded.get(field) for field in ("wave_max_abs", "wave_rmse", "wave_relative_rmse")]
    if any(value is not None and (not number(value) or value < 0) for value in errors):
        raise ValueError("Recorded waveform errors must be finite nonnegative numbers or null")
    comparable = same_size and old["sample_count"] > 0 and old["nonfinite"] == new["nonfinite"] == 0
    if not comparable and any(value is not None for value in errors):
        raise ValueError("Noncomparable recorded waveforms must not have numeric waveform errors")
    if comparable and any(value is None for value in errors[:2]):
        raise ValueError("Comparable recorded waveforms require max_abs and RMSE")
    if same_bits and comparable and (errors[0] != 0 or errors[1] != 0 or errors[2] not in (None, 0)):
        raise ValueError("Identical recorded waveforms cannot have nonzero waveform errors")
    return old, new, same_size, same_bits, *errors


def waveform(e2e, metric, speech, errors):
    if metric is None:
        return None, None
    relative = metric.get("output_directory")
    if not isinstance(relative, str):
        errors.append("A request is missing output_directory")
        return None, None
    path = e2e / relative / "waveform.f32"
    try:
        path.resolve().relative_to(e2e.resolve())
    except ValueError:
        errors.append(f"Unsafe output directory: {relative}")
        return None, None
    if not path.is_file():
        if speech:
            errors.append(f"Missing waveform: {relative}/waveform.f32")
        return None, None
    raw = path.read_bytes()
    identity = {"path": str(path.relative_to(e2e)), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "applicable_to_speech": speech}
    if len(raw) % 4:
        errors.append(f"Malformed float32 waveform: {relative}")
        identity["valid_float32_size"] = False
        return identity, None
    identity["valid_float32_size"] = True
    identity["sample_count"] = len(raw) // 4
    if not speech:
        if raw:
            errors.append(f"Text-only request unexpectedly has waveform bytes: {relative}")
        return identity, None
    samples = array.array("f")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    identity["nonfinite"] = sum(not math.isfinite(x) for x in samples)
    identity["matches_metric_samples"] = len(samples) == metric.get("wave_samples")
    return identity, samples


def compare(e2e, a, b, speech, errors, recorded=None):
    old_identity = new_identity = old_wave = new_wave = None
    if recorded is None:
        old_identity, old_wave = waveform(e2e, a, speech, errors)
        new_identity, new_wave = waveform(e2e, b, speech, errors)
    def tokens_equal(key):
        if a is None or b is None:
            return None
        left, right = a.get(key), b.get(key)
        if not isinstance(left, list) or not isinstance(right, list):
            errors.append(f"Missing token arrays for {key}")
            return None
        return left == right
    text_equal = tokens_equal("text_token_ids")
    speech_equal = tokens_equal("speech_token_ids") if speech else None
    output_valid = bool(a and b and a.get("success") is True and b.get("success") is True)
    wave_equal = wave_size_equal = max_abs = rmse = relative_rmse = None
    if speech:
        if recorded is not None:
            old_identity, new_identity, wave_size_equal, wave_equal, max_abs, rmse, relative_rmse = recorded_waveform_comparison(recorded, a, b)
        output_valid = output_valid and bool(old_identity and new_identity
            and old_identity.get("sample_count", 0) > 0 and new_identity.get("sample_count", 0) > 0
            and old_identity.get("nonfinite") == 0 and new_identity.get("nonfinite") == 0
            and old_identity.get("matches_metric_samples") and new_identity.get("matches_metric_samples"))
        if recorded is None and old_identity is not None and new_identity is not None:
            wave_size_equal = old_identity["bytes"] == new_identity["bytes"]
            wave_equal = wave_size_equal and old_identity["sha256"] == new_identity["sha256"]
        if old_wave is not None and new_wave is not None and len(old_wave) == len(new_wave) and old_wave:
            if all(math.isfinite(x) for x in old_wave) and all(math.isfinite(x) for x in new_wave):
                squared = math.fsum((float(x) - y) ** 2 for x, y in zip(old_wave, new_wave))
                energy = math.fsum(float(x) ** 2 for x in old_wave)
                max_abs = max(abs(float(x) - y) for x, y in zip(old_wave, new_wave))
                rmse = math.sqrt(squared / len(old_wave))
                relative_rmse = math.sqrt(squared / energy) if energy else None
    else:
        output_valid = output_valid and bool(a.get("text") and b.get("text")) if a and b else False
    matched = output_valid and text_equal is True and (not speech or (speech_equal is True and wave_size_equal is True))
    strict = matched and (not speech or wave_equal is True)
    return {"outputs_valid": output_valid, "text_tokens_equal": text_equal,
            "speech_tokens_equal": speech_equal, "wave_size_equal": wave_size_equal,
            "wave_bitwise_equal": wave_equal, "wave_max_abs": max_abs,
            "wave_rmse": rmse, "wave_relative_rmse": relative_rmse,
            "matched_generated_workload": matched, "strict_correctness_pass": strict,
            "old_waveform": old_identity, "new_waveform": new_identity,
            "wave_comparison_note": "not applicable to audio2text" if not speech else
                ("Historical recorded waveform validation; waveform bytes not re-read. " if recorded is not None else "")
                + "Different lengths are not aligned or truncated for numeric comparison"}


def group_summary(rows):
    timed = [x for x in rows if x["latency"]["old_over_new"] is not None]
    successful = [x for x in timed if x["comparison"]["outputs_valid"]]
    strict = [x for x in timed if x["comparison"]["strict_correctness_pass"]]
    def totals(values):
        if not values:
            return {"pairs": 0, **delta(None, None)}
        return {"pairs": len(values), **delta(sum(x["latency"]["old_s"] for x in values),
                                               sum(x["latency"]["new_s"] for x in values))}
    return {"expected_pairs": len(rows), "paired_timing_available": len(timed),
            "both_outputs_valid": sum(x["comparison"]["outputs_valid"] for x in rows),
            "matched_generated_workload_pairs": sum(x["comparison"]["matched_generated_workload"] for x in rows),
            "strict_equal_output_pairs": sum(x["comparison"]["strict_correctness_pass"] for x in rows),
            "all_available_system_pair_totals": totals(timed),
            "successful_system_pair_totals": totals(successful),
            "strict_equal_output_totals": totals(strict),
            "median_per_input_speed_ratio": statistics.median(x["latency"]["old_over_new"] for x in timed) if timed else None,
            "interpretation": "System-route elapsed time on the same selected inputs; output changes can alter generated work. This is not isolated-kernel speedup or task-quality equivalence."}


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def analyze(root, out, waveform_root=None, recorded_checks_path=None):
    if waveform_root is not None and recorded_checks_path is not None:
        raise ValueError("External raw waveforms and recorded waveform evidence are mutually exclusive")
    errors = []
    e2e = root / "e2e"
    wave_root = waveform_root if waveform_root is not None else e2e
    mode = ("recorded_evidence_not_rehashed" if recorded_checks_path is not None else
            "external_raw_waveforms_rehashed" if waveform_root is not None else "raw_waveforms_rehashed")
    run = load_json(e2e / "run.json", errors, required=True) or {}
    rows = lines(e2e / "results.jsonl", errors)
    runtime_path = next((root / name for name in ("records/process.txt", "records/process.log")
                         if (root / name).is_file()), root / "records/process.txt")
    runtime_affected = audit_runtime_log(runtime_path, rows, errors)
    declared = load_json(e2e / "comparisons.json", errors) or []
    manifest_paths = [root / "bundle/manifest.jsonl", root / "inputs/bundle/manifest.jsonl"]
    manifest = next((p for p in manifest_paths if p.is_file()), None)
    items = lines(manifest, errors) if manifest else []
    if manifest is None:
        errors.append("Missing frozen input manifest; expected samples cannot be established")
        seen = set()
        for value in rows:
            if not value.get("warmup") and value.get("row") not in seen:
                items.append({"original_row_number": value.get("row"), "index": value.get("id"),
                              "task_type": value.get("task_type"), "subset": value.get("subset")})
                seen.add(value.get("row"))
    formal = [x for x in rows if x.get("warmup") is False]
    indexed = {}
    for value in formal:
        key = (value.get("row"), value.get("pair"), value.get("variant"))
        if key in indexed:
            errors.append(f"Duplicate formal record key: {key}")
        else:
            indexed[key] = value
    checks = {}
    if isinstance(declared, list):
        for value in declared:
            if isinstance(value, dict) and value.get("warmup") is False:
                checks[(value.get("row"), value.get("pair"))] = value.get("paired")
    else:
        errors.append("comparisons.json must be an array")
    pair_count = run.get("pairs_per_sample", 1)
    if type(pair_count) is not int or pair_count < 1:
        errors.append("Invalid pairs_per_sample; reporting one expected pair per input")
        pair_count = 1
    recorded = None
    recorded_source = None
    if recorded_checks_path is not None:
        expected = {(item.get("original_row_number"), pair): item.get("index")
                    for item in items for pair in range(pair_count)}
        recorded, recorded_source = recorded_waveform_checks(root, recorded_checks_path, expected,
                                                             runtime_path, manifest)
    details, answers = [], []
    for ordinal, item in enumerate(items, 1):
        row, identifier = item.get("original_row_number"), item.get("index")
        for pair in range(pair_count):
            a, b = indexed.get((row, pair, "old")), indexed.get((row, pair, "new"))
            speech = item.get("task_type") != "audio2text"
            available = [name for name, value in (("old", a), ("new", b)) if value is not None]
            for name, value in (("old", a), ("new", b)):
                if value is not None and (value.get("id") != identifier or value.get("task_type") != item.get("task_type")):
                    if recorded is not None:
                        raise ValueError(f"Recorded waveform request identity differs from manifest: row={row}, pair={pair}, variant={name}")
                    errors.append(f"Record identity differs from manifest: row={row}, pair={pair}, variant={name}")
            comparison = compare(wave_root, a, b, speech, errors,
                                 recorded=recorded[(row, pair)] if recorded is not None else None)
            entry = {"sample_ordinal": ordinal, "row": row, "id": identifier, "pair": pair,
                     "task_type": item.get("task_type"), "subset": item.get("subset"),
                     "available_variants": available, "missing_variants": [x for x in ["old", "new"] if x not in available],
                     "latency": delta(a.get("request_s") if a else None, b.get("request_s") if b else None),
                     "ttfa": delta(a.get("ttfa_s") if a else None, b.get("ttfa_s") if b else None),
                     "audio_encoder": delta(a.get("audio_encoder_including_prepare_s") if a else None,
                                            b.get("audio_encoder_including_prepare_s") if b else None),
                     "comparison": comparison, "runner_declared_comparison": checks.get((row, pair)),
                     "old_metrics": a, "new_metrics": b}
            details.append(entry)
            answers.append({"row": row, "id": identifier, "pair": pair, "task_type": item.get("task_type"),
                            "source_jsonl": item.get("source_jsonl"), "source_line_number": item.get("source_line_number"),
                            "old_text": a.get("text") if a else None, "new_text": b.get("text") if b else None,
                            "text_equal": (a.get("text") == b.get("text")) if a and b else None,
                            "answer_quality_evaluated": False})
    expected_keys = {(x.get("original_row_number"), pair, variant) for x in items for pair in range(pair_count) for variant in ["old", "new"]}
    unexpected = [list(key) for key in indexed if key not in expected_keys]
    if unexpected:
        errors.append(f"Unexpected formal record keys: {unexpected}")
    complete = (run.get("state") == "completed" and bool(items)
                and len(indexed) == len(expected_keys) and not unexpected
                and all(not x["missing_variants"] for x in details))
    inputs = {}
    for path in [e2e / "run.json", e2e / "results.jsonl", e2e / "comparisons.json", runtime_path,
                 root / "records/command.json", root / "records/model_before.json"]:
        if path.is_file():
            inputs[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    if manifest:
        inputs[str(manifest.relative_to(root))] = {"bytes": manifest.stat().st_size, "sha256": sha(manifest)}
    summary = {"schema_version": 1, "comparison_kind": "system_route_A_B",
               "analysis_execution": "offline_reanalysis_not_device_benchmark",
               "wave_validation_mode": mode, "recorded_waveform_checks_source": recorded_source,
               "baseline": "Original MNN full audio encoder via unchanged mllm OpenCL runtime, old ConvTranspose",
               "candidate": "Existing RKNN Conv1-GELU-Conv2-GELU prefix plus CPU encoder tail and BigVGAN tile4",
               "run_state": run.get("state"), "retrieved_complete_input_pairs": complete,
               "expected_samples": len(items), "expected_pairs": len(items) * pair_count,
               "formal_records_found": len(formal), "warmup_records_found": len(rows) - len(formal),
               "all_samples": group_summary(details),
               "by_task_type": {kind: group_summary([x for x in details if x["task_type"] == kind])
                                for kind in sorted({str(x["task_type"]) for x in details})},
               "per_pair": details, "answers_file": "answers.json", "source_files": inputs,
               "runtime_invalid_requests": runtime_affected,
               "record_integrity_passed": complete and not errors,
               "audit_errors": errors, "audit_passed": complete and not errors and not runtime_affected,
               "limitations": [
                   "每条样本只有一对正式A/B，不报告稳定p95或统计显著性。",
                   "两个模型实例常驻、请求串行执行；本轮不证明单实例内存节省或并发吞吐。",
                   "每侧只预热一条语音任务；其他动态shape首次编码/编译成本保留在正式请求时间内。",
                   "模型源文件相同不等于数值执行相同：RKNN前缀使用FP16，可能改变embedding和生成输出。",
                   "文本/语音token和波形比较检查输出是否保持，不等于答案质量、语义或听感评测通过。",
                   "audio2text的空波形不作为语音正确性证据。",
                   "请求计时从response开始至文本返回或语音完整生成；模型加载、预热及输出文件落盘不计入。",
                   "阶段计数器可能重叠，不相加作为端到端耗时分解。",
                   "本次是系统路径对比，不证明存在独立优化的Conv2+GELU kernel，也不把收益分别归因于两个改动。"]}
    if recorded is not None:
        summary["limitations"].append("本次离线复算只校验已绑定的文本文件身份，复用历史设备波形校验记录；未读取波形字节，未重新计算波形误差，也不是新的板端测试。")
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_diagnostics.json").write_text(json.dumps(runtime_affected, ensure_ascii=False, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n")
    fields = ["sample_ordinal", "row", "id", "task_type", "subset", "pair", "old_s", "new_s", "saved_s", "old_over_new", "reduction_pct",
              "outputs_valid", "old_success", "new_success", "old_runner_reported_success", "new_runner_reported_success",
              "old_runtime_issues", "new_runtime_issues", "old_text_tokens", "new_text_tokens", "old_speech_tokens", "new_speech_tokens",
              "old_wave_samples", "new_wave_samples", "text_tokens_equal", "speech_tokens_equal", "wave_bitwise_equal",
              "matched_generated_workload", "strict_correctness_pass", "missing_variants"]
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in details:
            a, b = item["old_metrics"] or {}, item["new_metrics"] or {}
            writer.writerow({**{k: item[k] for k in ["sample_ordinal", "row", "id", "task_type", "subset", "pair"]},
                             **item["latency"], **{k: item["comparison"][k] for k in ["outputs_valid", "text_tokens_equal", "speech_tokens_equal", "wave_bitwise_equal", "matched_generated_workload", "strict_correctness_pass"]},
                             "old_success": a.get("success"), "new_success": b.get("success"),
                             "old_runner_reported_success": a.get("runner_reported_success"),
                             "new_runner_reported_success": b.get("runner_reported_success"),
                             "old_runtime_issues": " | ".join(d["message"] for d in a.get("runtime_diagnostics", [])),
                             "new_runtime_issues": " | ".join(d["message"] for d in b.get("runtime_diagnostics", [])),
                             "old_text_tokens": a.get("text_tokens"), "new_text_tokens": b.get("text_tokens"),
                             "old_speech_tokens": a.get("speech_tokens"), "new_speech_tokens": b.get("speech_tokens"),
                             "old_wave_samples": a.get("wave_samples"), "new_wave_samples": b.get("wave_samples"),
                             "missing_variants": ",".join(item["missing_variants"])})
    total = summary["all_samples"]["successful_system_pair_totals"]
    text = ["# Omni 组合路径端到端 A/B", "",
            f"离线分析；波形校验模式：`{mode}`。" + ("复用已绑定的历史设备波形校验记录，未读取或重新计算波形字节；时序、token与运行错误从原始文本记录复算。" if recorded is not None else "本次从原始波形文件重新计算身份与数值误差；不会运行模型或连接设备。"), "",
            f"运行状态：`{summary['run_state']}`；冻结输入 {len(items)} 条，取得 {len(formal)} 条正式请求记录；成对数据完整：{complete}。",
            "", "基线：原 MNN/OpenCL 完整音频编码器 + 旧 ConvTranspose。候选：既有 RKNN/NPU 音频前缀 + CPU 编码器尾部 + BigVGAN tile4。",
            "", f"音频特征与输出均有效的配对请求总时间：**{fmt(total['old_s'])} → {fmt(total['new_s'])} s**；系统时间比 **{fmt(total['old_over_new'])}×**，变化 **{fmt(total['reduction_pct'], 2)}%**。此值仅覆盖音频/输出有效的配对，不含编码失败样本，也不代表全部10条通过。",
            "", f"正式请求严格输出一致 {summary['all_samples']['strict_equal_output_pairs']}/{len(details)} 对；生成工作量一致 {summary['all_samples']['matched_generated_workload_pairs']}/{len(details)} 对。输出变化的配对只能描述系统路径耗时，不能当作同输出优化收益或答案质量通过。",
            "", "| 序号 | 样本 | 类型 | 旧 s | 新 s | 旧/新 | 测量有效 | 输出严格一致 | 缺失 |", "|---:|---|---|---:|---:|---:|---|---|---|"]
    for item in details:
        text.append(f"| {item['sample_ordinal']} | {item['id']} | {item['task_type']} | {fmt(item['latency']['old_s'])} | {fmt(item['latency']['new_s'])} | {fmt(item['latency']['old_over_new'])} | {item['comparison']['outputs_valid']} | {item['comparison']['strict_correctness_pass']} | {','.join(item['missing_variants']) or '—'} |")
    text += ["", "## 按任务类型汇总", "", "| 类型 | 配对数 | 旧总时间 s | 新总时间 s | 旧/新 | 严格同输出对数 |", "|---|---:|---:|---:|---:|---:|"]
    for kind, group in summary["by_task_type"].items():
        timed = group["successful_system_pair_totals"]
        text.append(f"| {kind} | {timed['pairs']} | {fmt(timed['old_s'])} | {fmt(timed['new_s'])} | {fmt(timed['old_over_new'])} | {group['strict_equal_output_pairs']} |")
    strict = summary["all_samples"]["strict_equal_output_totals"]
    text += ["", f"严格同输出子集：{strict['pairs']}对，耗时 {fmt(strict['old_s'])} → {fmt(strict['new_s'])} s，时间比 {fmt(strict['old_over_new'])}×；这只覆盖相应子集，不能推广到全部10条。"]
    if runtime_affected:
        text += ["", "## 音频编码失败：不得计为加速", "", "外层程序成功返回不代表音频特征提取成功。下列请求在完整日志中出现编码器shape错误/无音频特征，已从有效配对与严格一致性统计中剔除；原始记录和耗时仍完整保留。"]
        for bad in runtime_affected:
            text.append(f"- 样本{bad['row']} / {bad['variant']}：" + "；".join(d["message"] for d in bad["diagnostics"]))
    text += ["", "## 解释范围", "", *[f"- {s}" for s in summary["limitations"]], "", "## 产物", "", "- `summary.json`：分组统计、逐对记录、原始文件身份与审查问题。", "- `per_sample.csv`：所有预计配对，包括缺失与失败。", "- `answers.json`：新旧文本与来源位置，未执行任务质量评分。", ""]
    if errors:
        text += ["## 审查问题", "", *[f"- {x}" for x in errors], ""]
    (out / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return {k: summary[k] for k in ["run_state", "wave_validation_mode", "expected_samples", "formal_records_found", "retrieved_complete_input_pairs", "audit_passed", "audit_errors"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--waveform-root", type=Path,
                      help="External waveform directory with the same request paths relative to e2e")
    mode.add_argument("--recorded-waveform-checks", type=Path,
                      help="Explicit historical waveform evidence sidecar; does not re-read waveform bytes")
    args = parser.parse_args()
    try:
        result = analyze(args.root.expanduser().resolve(), args.out.expanduser().resolve(),
                         waveform_root=args.waveform_root.expanduser().resolve() if args.waveform_root else None,
                         recorded_checks_path=args.recorded_waveform_checks.expanduser().resolve() if args.recorded_waveform_checks else None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
