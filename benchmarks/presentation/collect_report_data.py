"""从冻结首轮原始证据生成中文汇报数据；不加载运行库、不推理、不下载。"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def require(condition, message):
    """使用显式异常，避免 python -O 跳过证据断言。"""
    if not condition:
        raise ValueError(message)


def validate_graph_integrity(graph):
    """从真实 inputs/outputs 独立重建边，拒绝数量相同但来源被改写的图。"""
    nodes, tensors = graph["nodes"], graph["tensors"]
    ids = [n["id"] for n in nodes]
    tids = [t["id"] for t in tensors]
    require(len(ids) == len(set(ids)), "节点 ID 重复")
    require(len(tids) == len(set(tids)), "tensor ID 重复")
    tensor_by_id = {t["id"]: t for t in tensors}
    producer, consumers = {}, defaultdict(set)
    for node in nodes:
        require(all(t in tensor_by_id for t in node["inputs"] + node["outputs"]), "张量引用越界")
        for tid in node["outputs"]:
            require(tid not in producer, "tensor producer 不唯一")
            producer[tid] = node["id"]
    expected = []
    for node in nodes:
        for slot, tid in enumerate(node["inputs"]):
            consumers[tid].add(node["id"])
            if tid in producer:
                expected.append((producer[tid], node["id"], tid, slot))
    actual = [(e["source"], e["target"], e["tensor_id"], e["input_slot"]) for e in graph["edges"]]
    require(Counter(expected) == Counter(actual), "图边与真实张量 producer/consumer 不一致")
    for tensor in tensors:
        tid = tensor["id"]
        require(tensor["producer"] == producer.get(tid), "tensor producer 声明错误")
        require(tensor["consumers"] == sorted(consumers[tid]), "tensor consumers 声明错误")
    order = graph["topological_order"]
    require(set(order) == set(ids) and len(order) == len(ids), "拓扑序未覆盖所有节点")
    position = {ident: index for index, ident in enumerate(order)}
    require(all(position[s] < position[t] for s, t, _, _ in expected), "图存在环或拓扑序错误")
    return {"node_count": len(nodes), "tensor_count": len(tensors), "edge_count": len(actual),
            "unique_producers": True, "edges_reconstructed_from_tensor_ids": True}


class Sources:
    """哈希对应本次真正读取的字节；JSON 仅保存路径、哈希与长度。"""
    def __init__(self, root):
        self.root, self.entries = Path(root).resolve(), {}

    def bytes(self, relative, expected=None):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "证据路径越出工作区")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = path.relative_to(self.root).as_posix()
        entry = {"sha256": digest, "bytes": len(data)}
        if expected is not None:
            require(digest == expected, f"证据 SHA-256 不一致：{key}")
        if key in self.entries:
            require(self.entries[key] == entry, f"汇总期间文件改变：{key}")
        self.entries[key] = entry
        return data

    def json(self, relative, expected=None):
        return json.loads(self.bytes(relative, expected).decode("utf-8-sig"))

    def jsonl(self, relative):
        return [json.loads(line) for line in self.bytes(relative).decode("utf-8-sig").splitlines() if line.strip()]

    def sha(self, relative, expected=None):
        self.bytes(relative, expected)
        return self.entries[(self.root / relative).resolve().relative_to(self.root).as_posix()]["sha256"]

    def collection_script(self, name, expected):
        """以采集记录的哈希解析源码，不用修复后的代码替换历史证据。"""
        require(isinstance(name, str) and name.endswith(".py")
                and "/" not in name and "\\" not in name, "采集源码文件名必须是 Python 文件 basename")
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "采集源码 SHA-256 格式错误")
        relative = "benchmarks/yolov8n_home/scripts/" + name
        current = (self.root / relative).resolve()
        require(current.is_relative_to(self.root), "采集源码路径越出工作区")
        if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            relative = f"benchmarks/fixtures/yolov8n_home_sources/{expected}/{name}"
        self.sha(relative, expected)
        return relative


def stats(values):
    """采用与 NumPy 默认 percentile 一致的线性分位数；不相加各节点 p95。"""
    ordered = sorted(float(v) for v in values)
    require(bool(ordered) and all(math.isfinite(v) and v >= 0 for v in ordered), "统计值必须是非负有限数")
    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = math.floor(index)
        hi = math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"count": len(ordered), "mean": math.fsum(ordered) / len(ordered),
            "p95": percentile(.95), "min": ordered[0], "max": ordered[-1],
            "median": percentile(.5)}


def ms_fields(value):
    return {"sample_count": value["count"], **{key + "_ms": value[key] for key in ("mean", "p95", "min", "max")}}


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), f"数值交叉核对失败：{label}: {actual} != {expected}")


def check_stats(actual, expected, label):
    require(actual["count"] == expected["count"], f"统计样本数不一致：{label}")
    for key in ("mean", "p95", "min", "max", "median"):
        close(actual[key], expected[key], label + "." + key)


def load_graph_tools(root):
    """按确切文件路径导入首轮接口，避免与并行的新模型脚本重名。"""
    path = root / "benchmarks/yolov8n_home/scripts/graph_tools.py"
    spec = importlib.util.spec_from_file_location("home_report_graph_tools", path)
    if spec is None or spec.loader is None:
        raise ValueError("无法导入首轮图工具")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_key(node):
    if node["type"] == "InputOperator":
        return "input"
    if node["type"] == "OutputOperator":
        return "output"
    matches = set(re.findall(r"/model\.(0|[1-9]\d*)(?=/|$)", node["name"]))
    require(len(matches) == 1, f"模块路径不唯一或缺失：{node['name']}")
    return "model." + matches.pop()


def semantic_stage(module):
    if module in ("input", "output"):
        return module.title()
    index = int(module.split(".")[1])
    require(0 <= index <= 22, "当前语义映射仅适用于已核验的 model.0–22")
    return "Backbone" if index <= 9 else "Neck" if index <= 21 else "Head"


def module_label(module):
    if module in ("input", "output"):
        return {"input": "输入接口", "output": "九路输出接口"}[module]
    index = int(module.split(".")[1])
    kind = ("SPPF" if index == 9 else "检测头" if index == 22 else
            "上采样" if index in (10, 13) else "拼接" if index in (11, 14, 17, 20) else
            "C2f" if index in (2, 4, 6, 8, 12, 15, 18, 21) else "卷积")
    return f"{module} · {kind}"


def fold_modules(graph):
    groups = defaultdict(list)
    node_module = {}
    for node in graph["nodes"]:
        key = module_key(node)
        groups[key].append(node["id"])
        node_module[node["id"]] = key
    pairs = defaultdict(list)
    for index, edge in enumerate(graph["edges"]):
        pair = node_module[edge["source"]], node_module[edge["target"]]
        if pair[0] != pair[1]:
            pairs[pair].append(index)
    edges = [{"source": s, "target": t,
              "tensor_ids": sorted({graph["edges"][i]["tensor_id"] for i in indices}),
              "node_edge_indices": indices, "node_edge_count": len(indices)}
             for (s, t), indices in pairs.items()]
    require(sum(len(ids) for ids in groups.values()) == len(graph["nodes"]), "模块未覆盖全部节点")
    require(set(groups) == {"input", "output"} | {f"model.{i}" for i in range(23)}, "模块集合不符")
    return dict(groups), node_module, edges


def boundary(graph, ids):
    nodes = {n["id"]: n for n in graph["nodes"]}
    tensors = {t["id"]: t for t in graph["tensors"]}
    members = set(ids)
    incoming = sorted({tid for ident in members for tid in nodes[ident]["inputs"] if tensors[tid]["producer"] not in members})
    outgoing = sorted({tid for ident in members for tid in nodes[ident]["outputs"]
                       if tensors[tid]["is_graph_output"] or set(tensors[tid]["consumers"]) - members})
    return {"input_tensor_ids": incoming, "output_tensor_ids": outgoing,
            "produced_input_tensor_ids": [tid for tid in incoming if tensors[tid]["producer"] is not None],
            "unproduced_input_tensor_ids": [tid for tid in incoming if tensors[tid]["producer"] is None]}


def resource_summary(start, completion, rows):
    snapshots = [start["resources"], *rows, completion["resources"]]
    require(all(s["frequency_locked"] is False for s in snapshots), "锁频声明与快照不一致")
    observed, governors = defaultdict(set), defaultdict(set)
    for snapshot in snapshots:
        for path, value in snapshot["sysfs"].items():
            if value is None:
                continue
            if path.endswith(("/temp", "/cur_freq", "/scaling_cur_freq")):
                observed[path].add(float(value))
            elif path.endswith(("/governor", "/scaling_governor")):
                governors[path].add(value)
    return {"frequency_locked": False, "snapshot_count": len(snapshots),
            "observed_values": {k: sorted(v) for k, v in observed.items()},
            "governors": {k: sorted(v) for k, v in governors.items()},
            "process_max_rss_kib": max(s["process_max_rss_kib"] for s in snapshots),
            "power_watts": None, "energy_joules": None,
            "note": "CPU/NPU/DDR 未锁频；离散快照不证明全程频率、温度恒定。CPU 频率单位 kHz，devfreq 为 Hz，温度为毫摄氏度。"}


def validate_measurements(rows, samples, repeats, mode):
    expected = {(s["id"], repeat) for s in samples for repeat in range(repeats)}
    actual = [(r["image_id"], r["repeat"]) for r in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected, f"{mode} 执行覆盖不完整或重复")
    sample_names = {s["id"]: s["sample_id"] for s in samples}
    residuals = []
    for row in rows:
        require(row["status"] == "success" and row["mode"] == mode, f"{mode} 状态/模式不一致")
        require(row["sample_id"] == sample_names[row["image_id"]], "sample_id 与图像 ID 不一致")
        require(row["clock_domain"] == "host_perf_counter_ns", "主机时钟口径改变")
        keys = ("start_ns", "preprocessed_ns", "inference_complete_ns", "end_ns")
        times = [row[key] for key in keys]
        require(times == sorted(times), "主机时间戳非单调")
        for field, left, right in (("preprocess_ms", 0, 1), ("inference_wall_ms", 1, 2), ("postprocess_ms", 2, 3), ("e2e_ms", 0, 3)):
            close(row[field], (times[right] - times[left]) / 1e6, mode + "." + field)
        summed = math.fsum(row[field] for field in ("preprocess_ms", "inference_wall_ms", "postprocess_ms"))
        close(summed, row["e2e_ms"], mode + ".阶段之和")
        residuals.append(abs(summed - row["e2e_ms"]))
    return {"expected": len(expected), "success": len(rows), "failed": 0,
            "max_stage_sum_error_ms": max(residuals)}


def collect(root):
    """只读全部证据，返回前端稳定结构；任何断言不成立则不输出新 JSON。"""
    root = Path(root).resolve()
    source = Sources(root)
    h = "benchmarks/yolov8n_home/"
    config = source.json(h + "config.json")
    manifest = source.json(h + "dataset/samples.json")
    summary = source.json(h + "summary.json")
    provenance = source.json(h + "models/provenance.json")
    graph = source.json(h + "graphs/compiled_graph.json")
    records = source.jsonl(h + "RK3588Bench.jsonl")
    tools = load_graph_tools(root)
    source.sha(h + "scripts/graph_tools.py")
    source.sha(h + "scripts/analyze_run.py")
    source.sha("benchmarks/presentation/collect_report_data.py")
    for kind, filename in (("onnx", "yolov8n.onnx"), ("rknn", "yolov8n_fp16.rknn")):
        source.sha(h + "models/" + filename, config[kind + "_sha256"])
        require(provenance[kind]["sha256"] == config[kind + "_sha256"], "模型来源哈希不一致")
    require(graph["model_sha256"] == config["rknn_sha256"], "图不属于首轮模型")
    graph_checks = validate_graph_integrity(graph)
    require((len(graph["nodes"]), len(graph["tensors"]), len(graph["edges"])) == (108, 257, 138), "首轮图数量改变")
    samples = manifest["samples"]
    ids = sorted(s["id"] for s in samples)
    require(len(ids) == len(set(ids)) == manifest["count"] == config["sampling"]["count"] == 40, "冻结样本不为40个唯一图像")
    by_bench = {r["source"]["image_id"]: r for r in records}
    require(len(records) == len(by_bench) == 40 and set(by_bench) == set(ids), "RK3588Bench 样本覆盖不符")
    native = tools.extract_graph(root / h / "models/yolov8n_fp16.rknn")
    saved_native = source.json(h + "graphs/native_graph.json")
    require(native["graphs"] == saved_native["graphs"], "实际 RKNN 与保存的 native 图不一致")
    run_rows, runs, count_checks = {}, {}, {}
    for mode, run_dir, repeats in (("baseline", "baseline_diagnostic_01", 10), ("detail", "detail_diagnostic_01", 1), ("verify", "verify_01", 1)):
        relative = h + "runs/" + run_dir + "/"
        rows = source.jsonl(relative + "measurements.jsonl")
        count_checks[mode] = validate_measurements(rows, samples, repeats, mode)
        start = source.json(relative + "start.json")
        completion = source.json(relative + "completion.json")
        runtime = source.json(relative + "runtime.json")
        warmup = source.json(relative + "warmup.json")
        resources = resource_summary(start, completion, source.jsonl(relative + "resources.jsonl"))
        require(start["config_sha256"] == source.entries[h + "config.json"]["sha256"], "运行配置哈希不一致")
        require(start["samples_sha256"] == source.entries[h + "dataset/samples.json"]["sha256"], "运行样本哈希不一致")
        require(completion["expected_measurements"] == len(rows) and completion["failed"] == 0, "completion 与原始记录不一致")
        require(runtime["metadata"]["detail"] == (mode == "detail") and runtime["metadata"]["core_mask"] == 7, "采集标志/核掩码不一致")
        require(len(warmup) == 3, "预热数量改变")
        collection_sources = {name: source.collection_script(name, digest)
                              for name, digest in start["source_sha256"].items()}
        for key in ("expected", "success", "failed"):
            require(count_checks[mode][key] == summary["counts"][mode][key], "summary 执行计数不一致")
        run_rows[mode] = rows
        runs[mode] = {"path": relative.rstrip("/"), "start_utc": start["resources"]["wall_clock_utc"],
                      "end_utc": completion["resources"]["wall_clock_utc"], "runtime": runtime,
                      "warmup_count": len(warmup), "environment": {k: v for k, v in start["environment"].items() if k != "background_processes"},
                      "resources": resources, "collection_source_files": collection_sources}
    stage_keys = ("preprocess_ms", "inference_wall_ms", "postprocess_ms", "e2e_ms")
    baseline_stages = {key: stats([r[key] for r in run_rows["baseline"]]) for key in stage_keys}
    baseline_api = {}
    for key in run_rows["baseline"][0]["api"]:
        value = stats([r["api"][key] for r in run_rows["baseline"]])
        check_stats(value, summary["baseline"]["pooled_repetitions"]["api"][key], "baseline.api." + key)
        if key.endswith("_us"):
            converted = {k: v if k == "count" else v / 1000 for k, v in value.items()}
            baseline_api[key[:-3] + "_ms"] = converted
        else:
            baseline_api[key] = value
    for key, value in baseline_stages.items():
        check_stats(value, summary["baseline"]["pooled_repetitions"][key], "baseline." + key)
    close(baseline_stages["e2e_ms"]["mean"], math.fsum(baseline_stages[k]["mean"] for k in stage_keys[:-1]), "baseline 总均值守恒")
    groups, node_module, module_edges = fold_modules(graph)
    detail_by_image = {r["image_id"]: r for r in run_rows["detail"]}
    raw_by_key = {(r["image_id"], r["mode"], r["repeat"]): r for mode in ("baseline", "detail") for r in run_rows[mode]}
    parsed_by_image, hotspot_by_image, detail_samples = {}, {}, []
    expected_identity = [(n["id"], n["name"], n["type"], n["target"]) for n in graph["nodes"]]
    report_paths = set()
    for image_id in ids:
        execution, bench = detail_by_image[image_id], by_bench[image_id]
        require(bench["model_sha256"] == config["rknn_sha256"], "Bench 模型身份不一致")
        require(bench["model_graph_sha256"] == source.entries[h + "graphs/compiled_graph.json"]["sha256"], "Bench 图哈希不一致")
        require(bench["status"] == "success" and bench["quality"]["status"] == "failed", "Bench 状态未保持首轮事实")
        require(len(bench["executions"]) == 11, "Bench 每图执行记录数量不符")
        bench_keys = []
        for record in bench["executions"]:
            key = (record["image_id"], record["mode"], record["repeat"])
            bench_keys.append(key)
            require(key in raw_by_key, "Bench 引用不存在的执行记录")
            require({k: v for k, v in record.items() if k != "nodes"} == raw_by_key[key], "Bench 执行记录与原始 JSONL 不一致")
        require(len(set(bench_keys)) == len(bench_keys), "Bench 执行记录重复")
        path = h + execution["perf"]
        source.sha(path, execution["perf_sha256"])
        report_paths.add(path)
        parsed = tools.parse_perf(root / path)
        bound = tools.bind_graph(native, parsed)
        require([(n["id"], n["name"], n["type"], n["target"]) for n in parsed] == expected_identity, "40份报告节点身份/设备映射不一致")
        for key in ("edges", "tensors", "inputs", "outputs", "topological_order"):
            require(bound[key] == graph[key], "逐报告图映射不一致：" + key)
        for fresh, saved in zip(bound["nodes"], graph["nodes"]):
            for key in ("index", "id", "name", "type", "target", "inputs", "outputs"):
                require(fresh[key] == saved[key], "实际模型与保存的编译节点不一致：" + key)
        parsed_by_image[image_id] = {n["id"]: n for n in parsed}
        bench_detail = next(r for r in bench["executions"] if r["mode"] == "detail")
        saved_rows = source.json(h + bench_detail["nodes"]["path"], bench_detail["nodes"]["sha256"])
        require(len(saved_rows) == len(parsed), "保存的节点报告数量不符")
        for left, right in zip(parsed, saved_rows):
            for key in ("id", "name", "type", "target", "dtype", "input_shape", "output_shape", "time_us"):
                require(left[key] == right[key], "原始 PERF 与保存的节点数据不一致：" + key)
        selected = tools.select_hotspots(bound, parsed)
        require(len(selected) == len(bench["hotspots"]) == 4, "热点数量不符")
        seen = set()
        for new, old in zip(selected, bench["hotspots"]):
            for key in ("name", "node_ids", "input_tensor_ids", "output_tensor_ids", "rank", "time_us", "time_ms", "replayable", "subgraph_io_available"):
                require(new[key] == old[key], "热点与 RK3588Bench 原始记录不一致：" + key)
            require(not (seen & set(new["node_ids"])), "同图热点重叠")
            seen.update(new["node_ids"])
            bounds = boundary(graph, new["node_ids"])
            require(new["input_tensor_ids"] == bounds["input_tensor_ids"] and new["output_tensor_ids"] == bounds["output_tensor_ids"], "热点边界不匹配")
            pending, visited = [new["node_ids"][0]], set()
            members = set(new["node_ids"])
            while pending:
                ident = pending.pop()
                if ident in visited:
                    continue
                visited.add(ident)
                for edge in graph["edges"]:
                    if edge["source"] == ident and edge["target"] in members:
                        pending.append(edge["target"])
                    if edge["target"] == ident and edge["source"] in members:
                        pending.append(edge["source"])
            require(visited == members, "热点不是连通分量")
        hotspot_by_image[image_id] = selected
        detail_samples.append({"image_id": image_id, "sample_id": execution["sample_id"], "perf_source": path,
                               "node_time_sum_ms": math.fsum(n["time_us"] for n in parsed) / 1000,
                               "target_ms": {target: math.fsum(n["time_us"] for n in parsed if n["target"] == target) / 1000 for target in ("CPU", "NPU")}})
    actual_reports = {(root / h / "runs/detail_diagnostic_01/perf" / p.name).relative_to(root).as_posix()
                      for p in (root / h / "runs/detail_diagnostic_01/perf").glob("[0-9]*.perf.txt")}
    require(report_paths == actual_reports and len(report_paths) == 40, "原始详细报告集合不完整")
    def group_stats(node_ids):
        return stats([math.fsum(parsed_by_image[image_id][ident]["time_us"] for ident in node_ids) / 1000 for image_id in ids])
    total = group_stats([n["id"] for n in graph["nodes"]])
    check_stats(total, summary["detail"]["node_time_sum_ms"], "detail.total")
    nodes = [{**{k: n[k] for k in ("id", "name", "type", "target", "inputs", "outputs")},
              "module": node_module[n["id"]], "stage": semantic_stage(node_module[n["id"]]),
              "dtype": parsed_by_image[ids[0]][n["id"]]["dtype"], **ms_fields(group_stats([n["id"]]))} for n in graph["nodes"]]
    modules = [{"id": key, "label": module_label(key), "stage": semantic_stage(key),
                "node_ids": groups[key], "node_count": len(groups[key]), **ms_fields(group_stats(groups[key])),
                **boundary(graph, groups[key])} for key in ["input", *[f"model.{i}" for i in range(23)], "output"]]
    type_stats, target_stats, stage_stats = [], [], []
    for field, output, summary_key in (("type", type_stats, "type_time_ms"), ("target", target_stats, "target_time_ms"), ("stage", stage_stats, None)):
        for value in dict.fromkeys(n[field] for n in nodes):
            member_ids = [n["id"] for n in nodes if n[field] == value]
            result = group_stats(member_ids)
            if summary_key:
                check_stats(result, summary["detail"][summary_key][value], "detail." + value)
                require(len(member_ids) == summary["detail"][field + "_counts"][value], "类型/设备节点数不符")
            output.append({field: value, "node_count": len(member_ids), "node_ids": member_ids,
                           "share_pct": result["mean"] / total["mean"] * 100, **ms_fields(result)})
    for group_name, collection in (("nodes", nodes), ("modules", modules), ("types", type_stats), ("targets", target_stats), ("stages", stage_stats)):
        close(math.fsum(item["mean_ms"] for item in collection), total["mean"], "均值守恒." + group_name)
    require(graph["validated_reports"] == len(ids), "保存图的验证报告计数不符")
    for key, count in (("report_count", 40), ("node_count", len(nodes)), ("edge_count", len(graph["edges"])), ("tensor_count", len(graph["tensors"]))):
        require(summary["detail"][key] == count, "summary 图数量不符")
    cross_edge_count = sum(node_module[e["source"]] != node_module[e["target"]] for e in graph["edges"])
    require(sum(e["node_edge_count"] for e in module_edges) == cross_edge_count, "模块边折叠遗漏跨模块原始边")
    hotspots = collect_hotspots(graph, hotspot_by_image, detail_samples, parsed_by_image, summary)
    datasets = collect_datasets(source, manifest, config)
    quality = collect_quality(source, manifest, config, summary, tools)
    onnx_info = collect_onnx_identity(root / h / "models/yolov8n.onnx")
    return {
        "meta": {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "workspace_root": root.as_posix(), "path_base": "所有来源路径均相对 workspace_root",
                 "run_id": config["run_id"], "experiment_date": runs["baseline"]["start_utc"][:10],
                 "model": {"name": config["model"], "task": config["task"], "target": config["target"],
                           "precision": config["precision"], "rknn_sha256": config["rknn_sha256"], "onnx_sha256": config["onnx_sha256"],
                           "model_zoo_commit": config["model_zoo_commit"], "conversion_recorded_at": provenance["recorded_at"],
                           "compiler": graph["metadata"]["compiler"], "source_model_url": provenance["model_download_url"],
                           "sdk_version": runs["baseline"]["runtime"]["metadata"]["sdk_version"], **onnx_info},
                 "preprocessing": {**config["input"], "decode": "读取原图字节，OpenCV imdecode 为 BGR uint8",
                                   "letterbox": "按 min(640/w,640/h) 线性缩放；居中补114，使用每张保存的实际缩放与补边参数",
                                   "normalization": "应用提交 RGB uint8 HWC；RKNN mean=0/std=255；参考 ONNX 为 float32 NCHW /255",
                                   "source": h + "scripts/yolo_common.py"},
                 "postprocessing": {"steps": ["三尺度分类置信度筛选", "四边×16 bins DFL softmax/期望", "逐类别 NMS", "按真实 scale_x/scale_y 与 pad 还原并裁剪坐标"],
                                    "confidence": config["baseline"]["confidence"], "nms_iou": config["baseline"]["nms_iou"],
                                    "max_detections": config["baseline"]["max_detections"], "score_sum_note": "九路中的 score-sum 不是 objectness；当前后处理按类别最大概率筛选"},
                 "time_semantics": {"unit": "ms，perf 原始 Time(us) 除以1000", "statistics": "mean=算术平均；p95=排序后 (n-1)*0.95 线性插值；各组先逐样本求和再计算分位数",
                                    "baseline": "400次=40图×10轮的混合分布；p95 不等于某一张图的稳定尾延迟",
                                    "detail": "40图各一次；节点/模块/类型/阶段/热点时间均为编译节点耗时的派生求和，不是墙钟或设备关键路径",
                                    "api": "C++ steady_clock 计 API 包围时间；Python perf_counter_ns 计输出复制、会话与外层阶段墙钟；perf_run 原值为微秒",
                                    "e2e": config["timing"]["e2e"], "baseline_excluded": config["timing"]["excluded"],
                                    "synchronization": config["timing"]["synchronization"], "cache": config["timing"]["cache"],
                                    "detail_wall_caveat": "run_board.py 在 session.run 外计时，详细轮的性能查询/报告写盘位于 session.run 内；详细轮墙钟包含此开销，不能照抄 config 中排除该项的声明",
                                    "cpu_node_note": "首轮CPU节点总计只含编译图Input/Output接口，不含应用预处理、后处理及所有运行库CPU开销",
                                    "fusion_note": "ConvExSwish/ConvSigmoid 等融合节点不能拆出未采集的内部算子独立时间"},
                 "semantic_mapping": {"method": "除 InputOperator/OutputOperator 单列外，以 FullName 中确切 /model.N/ 路径分组，不按节点编号区间",
                                      "stage_ranges": {"Input": "InputOperator", "Backbone": "model.0–9", "Neck": "model.10–21", "Head": "model.22", "Output": "OutputOperator"},
                                      "basis": "结合同一 ONNX 的模块路径、算子模式与真实连接：model.9 三次 MaxPool+Concat 为 SPPF；10/13 Resize 与跨尺度 Concat/卷积为特征融合；22 的 cv2/cv3 和九路输出为检测头",
                                      "boundary": "Backbone/Neck/Head、C2f/SPPF 是汇报语义注释，不是 RKNN 导出的阶段字段；上游某些配置将10–22统称head。只对本哈希模型适用。语义不生成边。",
                                      "evidence": [h + "models/yolov8n.onnx", h + "graphs/compiled_graph.json"],
                                      "module_edge_rule": "仅折叠实际 node edge 的模块端点；保留 tensor_ids 与原始 node_edge_indices（0起始）",
                                      "tensor_note": "未生产张量保留原始 kind/data_type；不根据 shape 或相邻编号推测常量、权重或依赖"},
                 "sources": dict(sorted(source.entries.items()))},
        "baseline": {"execution_count": len(run_rows["baseline"]), "sample_count": len(ids), "repeats_per_sample": 10,
                     "stages_ms": baseline_stages, "api_ms": baseline_api, "frequency_locked": False,
                     "resident": config["baseline"]["resident"], "sample_order_seed": config["baseline"]["sample_order_seed"],
                     **runs["baseline"], "performance_scope": "diagnostic_only", "formal_acceptance": False,
                     "per_sample": [{"image_id": image_id, "inference_wall_ms": stats([r["inference_wall_ms"] for r in run_rows["baseline"] if r["image_id"] == image_id]),
                                     "e2e_ms": stats([r["e2e_ms"] for r in run_rows["baseline"] if r["image_id"] == image_id])} for image_id in ids]},
        "detail": {"report_count": len(ids), "node_count": len(nodes), "tensor_count": len(graph["tensors"]), "edge_count": len(graph["edges"]),
                   "model_sha256": config["rknn_sha256"], "node_time_sum_ms": total, "nodes": nodes,
                   "edges": graph["edges"], "tensors": graph["tensors"], "inputs": graph["inputs"], "outputs": graph["outputs"],
                   "topological_order": graph["topological_order"], "modules": modules, "module_edges": module_edges,
                   "type_stats": sorted(type_stats, key=lambda x: -x["mean_ms"]), "target_stats": target_stats, "stage_stats": stage_stats,
                   "samples": detail_samples, "run": runs["detail"],
                   "time_semantics": "sum_node_times_not_wall_clock", "replayable": False},
        "hotspots": hotspots, "datasets": datasets, "quality": quality,
        "validation": {"status": "passed", "graph": graph_checks, "runs": count_checks,
                       "strictly_mapped_reports": len(ids), "strictly_mapped_node_rows": sum(len(v) for v in parsed_by_image.values()),
                       "module_count": len(modules), "module_edge_count": len(module_edges), "cross_module_node_edge_count": cross_edge_count,
                       "module_partition_covers_nodes_once": True, "hotspots_disjoint_connected_and_boundaries_verified": True,
                       "rk3588bench_executions_and_hotspots_match_raw": True, "actual_rknn_matches_saved_native_and_compiled_graph": True,
                       "summary_type_target_total_statistics_match": True, "means_conserved_across_nodes_types_modules_targets_stages": True,
                       "detail_mean_ms": total["mean"], "e2e_equals_preprocess_inference_postprocess": True,
                       "baseline_mean_e2e_ms": baseline_stages["e2e_ms"]["mean"], "hash_source_count": len(source.entries),
                       "scope": "验证现有文件和派生统计，未启动板端推理/下载；未重新读取全部预测张量或重算 COCO AP，质量任务指标引用已哈希的原始验证文件；通用数据下载日志未转为最终审计。"}}


def collect_hotspots(graph, selections, sample_rows, parsed, summary):
    """每图独立选四个；总体统计同时展示入选时与全部40图的耗时。"""
    by_name = defaultdict(list)
    for image_id, chosen in selections.items():
        for item in chosen:
            by_name[item["name"]].append((image_id, item))
    original = {h["name"]: h for h in summary["detail"]["selected_hotspots"]}
    require(set(original) == set(by_name), "热点候选名称与 summary 不一致")
    aggregated = []
    for name, pairs in by_name.items():
        exemplar = pairs[0][1]
        member_ids = exemplar["node_ids"]
        require(all(h["node_ids"] == member_ids for _, h in pairs), "同名热点节点集合不一致")
        selected_stats = stats([h["time_ms"] for _, h in pairs])
        all_stats = stats([math.fsum(rows[n]["time_us"] for n in member_ids) / 1000 for rows in parsed.values()])
        require(len(pairs) == original[name]["selected_sample_count"], "热点入选次数不一致")
        check_stats(selected_stats, original[name]["time_ms_when_selected"], "热点." + name)
        aggregated.append({"name": name, "node_ids": member_ids, "selected_sample_count": len(pairs),
                           "selection_fraction": len(pairs) / len(selections),
                           "rank_counts": dict(sorted(Counter(h["rank"] for _, h in pairs).items())),
                           "time_ms_when_selected": selected_stats, "time_ms_all_samples": all_stats})
    selection_count = sum(len(v) for v in selections.values())
    require(selection_count == summary["detail"]["hotspot_count"], "热点总入选数不一致")
    median = stats([s["node_time_sum_ms"] for s in sample_rows])["median"]
    representative = min(sample_rows, key=lambda s: (abs(s["node_time_sum_ms"] - median), s["image_id"]))
    fields = ("name", "rank", "node_ids", "input_tensor_ids", "output_tensor_ids", "time_ms", "replayable", "subgraph_io_available")
    return {"sample_count": len(selections), "selection_count": selection_count, "max_per_sample": 4,
            "method": "每图按精确模块路径的诱导图连通分量选节点耗时和前四；同图不重叠，跨图入选名单允许改变",
            "time_semantics": "sum_node_times_not_wall_clock", "replayable": False, "subgraph_io_available": False,
            "boundary_note": "边界仅为真实 tensor ID，含无producer张量；未捕获张量值，不能据此宣称可独立重放",
            "selection_stats": sorted(aggregated, key=lambda s: (-s["selected_sample_count"], -s["time_ms_when_selected"]["mean"])),
            "per_sample": [{"image_id": image_id, "selected": [{k: h[k] for k in ("name", "rank", "time_ms")} for h in chosen]}
                           for image_id, chosen in selections.items()],
            "representative": {"image_id": representative["image_id"], "sample_id": representative["sample_id"],
                               "selection_rule": "选择节点总耗时最接近40图中位数的样本；相同距离按 image_id 升序",
                               "node_time_sum_ms": representative["node_time_sum_ms"], "perf_source": representative["perf_source"],
                               "hotspots": [{k: h[k] for k in fields} for h in selections[representative["image_id"]]]}}


def collect_onnx_identity(path):
    import onnx
    model = onnx.load(path)
    outputs = []
    producers = {out: node for node in model.graph.node for out in node.output}
    for index, output in enumerate(model.graph.output):
        node = producers[output.name]
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        outputs.append({"index": index, "name": output.name, "shape": shape,
                        "producer_name": node.name, "producer_type": node.op_type,
                        "role": ("回归64通道/4×16bins", "80类Sigmoid概率", "类别概率和截断到[0,1]")[index % 3]})
    require(len(outputs) == 9 and all(o["producer_name"].startswith("/model.22/") for o in outputs), "ONNX 输出契约不符")
    require(sum(n.op_type == "MaxPool" and n.name.startswith("/model.9/") for n in model.graph.node) == 3, "SPPF语义依据不符")
    require({n.name.split("/")[1] for n in model.graph.node if n.op_type == "Resize"} == {"model.10", "model.13"}, "Neck上采样语义依据不符")
    return {"onnx_node_count": len(model.graph.node), "onnx_producer": {"name": model.producer_name, "version": model.producer_version},
            "onnx_opsets": [{"domain": entry.domain, "version": entry.version} for entry in model.opset_import],
            "batch": 1, "class_count": 80, "output_count": len(outputs), "outputs": outputs}


def collect_datasets(source, manifest, config):
    h, g = "benchmarks/yolov8n_home/", "benchmarks/yolov8n_general/"
    scenes = source.json(h + "dataset/scene_review.json")
    scene_by_id = {row["image_id"]: row for row in scenes["samples"]}
    home_ids = {s["id"] for s in manifest["samples"]}
    require(len(scene_by_id) == len(scenes["samples"]) and set(scene_by_id) == home_ids, "场景复核覆盖不一致")
    scene_counts = dict(Counter(s["setting"] for s in scenes["samples"]))
    require(scene_counts == {"室内": 35, "室外": 5}, "场景复核计数与首轮约定不符")
    annotation = source.json(h + manifest["subset_annotation"], manifest["subset_annotation_sha256"])
    categories = {c["id"]: c for c in annotation["categories"]}
    strata = Counter(s["stratum_category_id"] for s in manifest["samples"])
    require(set(strata) == set(manifest["category_order"]) and len(strata) == 8, "首轮分层覆盖不一致")
    require(all(count == manifest["per_category"] for count in strata.values()), "首轮每层样本数量不一致")
    sample_list = []
    for sample in manifest["samples"]:
        source.sha(h + sample["image"], sample["image_sha256"])
        source.sha(h + sample["input"], sample["input_sha256"])
        sample_list.append({"image_id": sample["id"], "sample_id": sample["sample_id"], "image": h + sample["image"],
                            "input": h + sample["input"], "width": sample["width"], "height": sample["height"],
                            "stratum": sample["stratum_name"], "setting": scene_by_id[sample["id"]]["setting"],
                            "observation": scene_by_id[sample["id"]]["observation"], "transform": sample["transform"]})
    split = source.json(g + "dataset/split.json")
    full_annotation = source.json(g + "dataset/instances_val2017.json", split["annotation_sha256"])
    dev_ids, acceptance_ids = set(split["dev_ids"]), set(split["acceptance_ids"])
    profile_ids = [p["id"] for p in split["profile_samples"]]
    all_ids = {im["id"] for im in full_annotation["images"]}
    require(dev_ids == home_ids and len(split["dev_ids"]) == 40, "通用开发集未沿用首轮40图")
    require(len(split["acceptance_ids"]) == len(acceptance_ids) == 4960, "通用验收集数量不符")
    require(not (dev_ids & acceptance_ids) and dev_ids | acceptance_ids == all_ids, "通用拆分不完整或重叠")
    require(len(profile_ids) == len(set(profile_ids)) == 80 and set(profile_ids) <= acceptance_ids, "80性能子集不属于独立验收集")
    require(split["development_manifest_sha256"] == source.entries[h + "dataset/samples.json"]["sha256"], "通用开发清单哈希不一致")
    profile_annotations = [a for a in full_annotation["annotations"] if a["image_id"] in set(profile_ids)]
    annotated = {a["image_id"] for a in full_annotation["annotations"]}
    rows = source.jsonl(g + "dataset/download_progress.jsonl")
    latest = {}
    for row in rows:
        require(row["id"] in all_ids, "下载日志包含拆分外图像")
        require(row["split_sha256"] == source.entries[g + "dataset/split.json"]["sha256"], "下载日志不属于当前冻结拆分")
        latest[row["id"]] = row
    successful_ids = {ident for ident, row in latest.items() if row["status"] == "ok"}
    missing_paths = [g + relative for relative in ("evidence/data_audit.json", "dataset/dev_samples.json", "dataset/acceptance_samples.json", "dataset/samples.json", "dataset/image_input_hashes.json")
                     if not (source.root / g / relative).exists()]
    # 最终审计文件若后来产生，应重新审阅其语义；当前汇报只呈现增量日志状态。
    final_audit_exists = (source.root / g / "evidence/data_audit.json").exists()
    final_audit = source.json(g + "evidence/data_audit.json") if final_audit_exists else None
    for item in split["strata"]:
        require(sum(p["stratum_name"] == item["name"] for p in split["profile_samples"]) == split["per_stratum"], "80子集分层配额错误")
    return {"home": {"dataset": manifest["dataset"], "sample_count": len(home_ids), "seed": manifest["seed"],
                     "strata": [{"category_id": ident, "name": categories[ident]["name"], "count": strata[ident]} for ident in manifest["category_order"]],
                     "scene_counts": scene_counts, "scene_method": scenes["method"], "real_household_domain_confirmed": scenes["real_household_domain_confirmed"],
                     "scene_note": "八类主分层每类5张是家庭相关物品子集；35室内/5室外不证明真实家庭域归属",
                     "sampling_definition": manifest["sampling_definition"], "license_id": config["sampling"]["license_id"],
                     "annotation_count": len(annotation["annotations"]), "category_count_in_annotations": len({a["category_id"] for a in annotation["annotations"]}),
                     "evaluation_category_count": len(categories), "samples": sample_list},
            "general": {"dataset": split["dataset"], "all_image_count": len(all_ids), "class_count": len(full_annotation["categories"]),
                        "development_count": len(dev_ids), "independent_acceptance_count": len(acceptance_ids),
                        "independent_acceptance_status": "未正式验收，当前交付仅40图开发结果", "development_disjoint_from_acceptance": True,
                        "acceptance_zero_annotation_count": len(acceptance_ids - annotated),
                        "profile_subset": {"count": len(profile_ids), "seed": split["seed"], "strata": split["strata"],
                                           "per_stratum": split["per_stratum"], "samples": split["profile_samples"],
                                           "actual_category_count": len({a["category_id"] for a in profile_annotations}), "annotation_count": len(profile_annotations),
                                           "status": "ID已冻结；没有新候选80图的正式正常/详细性能结果", "subset_of_acceptance": True,
                                           "download_log_success_count": len(set(profile_ids) & successful_ids)},
                        "download": {"log_row_count": len(rows), "unique_image_count": len(latest), "duplicate_record_count": len(rows) - len(latest),
                                     "latest_status_counts": dict(Counter(r["status"] for r in latest.values())), "logged_success_count": len(successful_ids),
                                     "development_logged_success_count": len(dev_ids & successful_ids), "acceptance_logged_success_count": len(acceptance_ids & successful_ids),
                                     "without_success_log_count": len(all_ids - successful_ids), "latest_verified_at_utc": max(r["verified_at_utc"] for r in rows),
                                     "final_data_audit_present": final_audit_exists, "final_data_audit_status": final_audit.get("status") if isinstance(final_audit, dict) else None,
                                     "final_manifests_missing": missing_paths, "audit_performed_by_this_generator": False,
                                     "note": "成功数量来自冻结下载日志，不代表本次逐图哈希/解码审计通过；缺少最终清单时不能称5000图数据就绪"},
                        "sampling_definition": split["sampling_definition"], "scene_scope": split["scene_scope"]}}


def collect_quality(source, manifest, config, summary, tools):
    h, g = "benchmarks/yolov8n_home/", "benchmarks/yolov8n_general/"
    verified = source.json(h + "runs/verify_01/verification.json")
    require(verified == summary["quality"], "首轮 summary 质量结论与原始验证文件不一致")
    for name, digest in verified["evidence"].items():
        source.sha(h + "runs/verify_01/" + name, digest)
    rows = source.json(h + "runs/verify_01/verification_samples.json")
    expected_ids = {s["id"] for s in manifest["samples"]}
    require(len(rows) == 40 and {r["image_id"] for r in rows} == expected_ids, "首轮质量样本覆盖错误")
    tensors = [t for row in rows for t in row["tensors"]["tensors"]]
    max_l2 = max(t["relative_l2"] for t in tensors)
    passed_images = sum(all(t["finite"] and t["relative_l2"] <= config["quality_gate_declared_before_inference"]["per_tensor_relative_l2_max"]
                            for t in row["tensors"]["tensors"]) for row in rows)
    close(max_l2, verified["max_relative_l2"], "首轮最大相对L2")
    require(len(tensors) == verified["tensor_count"] == 360, "首轮质量张量数量不符")
    matching: dict[str, int | float] = {key: sum(row["detection_matching"][key] for row in rows) for key in ("reference_count", "actual_count", "matched")}
    matching["reference_fraction"] = matching["matched"] / matching["reference_count"]
    matching["actual_fraction"] = matching["matched"] / matching["actual_count"]
    close(matching["reference_fraction"], verified["reference_match_fraction"], "首轮检测匹配率")
    require(verified["status"] == "failed" and passed_images < len(rows), "首轮失败结论被覆盖")
    ap_drop = verified["evaluations"]["onnx"]["AP"] - verified["evaluations"]["rknn"]["AP"]
    close(ap_drop, verified["AP_drop_absolute"], "首轮AP差")
    home = {"status": verified["status"], "sample_count": len(rows), "tensor_count": len(tensors),
            "numeric_passed_images": passed_images, "max_relative_l2": max_l2,
            "numeric_passed_tensors": sum(t["passed"] for t in tensors), "all_outputs_finite": all(t["finite"] for t in tensors),
            "gates": verified["gates"], "checks": verified["checks"], "detection_matching": matching,
            "evaluations": {backend: {k: v for k, v in result.items() if k != "all_stats"} for backend, result in verified["evaluations"].items()},
            "AP_drop_absolute": ap_drop, "performance_scope": "diagnostic_only", "formal_acceptance": False,
            "note": "首轮逐张量相对L2门槛失败；检测匹配率和子集AP差通过不能替代失败门槛。AP来自原验证文件，本生成器未重算COCOeval。",
            "source": h + "runs/verify_01/verification.json"}
    provenance = source.json(g + "models/model_provenance.json")
    result = source.json(g + "evidence/model/development40/development_results.json", provenance["development"]["result_sha256"])
    protocol = source.json(g + "quality_protocol.json")
    require(result["protocol_sha256"] == source.entries[g + "quality_protocol.json"]["sha256"], "开发门槛协议哈希不一致")
    source.sha(g + "evidence/model/quality_protocol.json", result["protocol_sha256"])
    audit = source.json(g + "evidence/model/artifact_audit.json")
    dev_rows = source.jsonl(g + "evidence/model/development40/samples.jsonl")
    require(len(dev_rows) == len({r["image_id"] for r in dev_rows}) == 40 and {r["image_id"] for r in dev_rows} == expected_ids, "开发40图覆盖错误")
    require(all(r["status"] == "success" for r in dev_rows), "开发图执行失败")
    input_sha = {s["id"]: s["input_sha256"] for s in manifest["samples"]}
    require(all(r["input_sha256"] == input_sha[r["image_id"]] for r in dev_rows), "开发图输入哈希与冻结首轮不一致")
    require(result["manifest_sha256"] == source.entries[h + "dataset/samples.json"]["sha256"], "开发结果未绑定冻结清单")
    for kind in ("onnx", "rknn"):
        require(result[kind + "_sha256"] == provenance[kind]["sha256"], "开发模型身份不符")
        source.sha(g + provenance[kind]["path"], provenance[kind]["sha256"])
    require(result["onnx_sha256"] == config["onnx_sha256"] and result["rknn_sha256"] != config["rknn_sha256"], "新旧模型身份隔离失败")
    gates = protocol["tensor_gates"]
    by_role = defaultdict(list)
    numeric_images, strict_images, numeric_tensors, strict_tensors = 0, 0, 0, 0
    for row in dev_rows:
        checks, strict = [], []
        require(len(row["numeric"]["tensors"]) == 9, "开发图不是九路输出")
        for tensor in row["numeric"]["tensors"]:
            role = tensor["role"]
            by_role[role].append(tensor)
            if role == "regression":
                passed = tensor["relative_l2"] <= gates["regression_relative_l2_max"]
            elif role == "probability":
                passed = (tensor["rmse"] <= gates["probability_rmse_max"] and
                          gates["probability_range"][0] <= tensor["actual_min"] <= tensor["actual_max"] <= gates["probability_range"][1])
            elif role == "clip_sum":
                passed = (tensor["mae"] <= gates["score_sum_mae_max"] and 0 <= tensor["actual_min"] <= tensor["actual_max"] <= 1)
            else:
                raise ValueError("未知输出角色")
            passed = passed and tensor["finite"]
            old_passed = tensor["finite"] and tensor["relative_l2"] <= .02
            require(passed == tensor["passed"] and old_passed == tensor["strict_2pct_passed"], "开发逐张量门槛重算不一致")
            checks.append(passed)
            strict.append(old_passed)
        numeric_images += all(checks)
        strict_images += all(strict)
        numeric_tensors += sum(checks)
        strict_tensors += sum(strict)
    for key, computed in (("numeric_passed_images", numeric_images), ("strict_2pct_passed_images", strict_images),
                          ("numeric_passed_tensors", numeric_tensors), ("strict_2pct_passed_tensors", strict_tensors),
                          ("sample_count", len(dev_rows)), ("success_count", len(dev_rows))):
        require(computed == result[key], "开发统计与逐图原始数据不一致：" + key)
    per_role = {}
    for role, items in by_role.items():
        per_role[role] = {"tensor_count": len(items), "max_relative_l2": max(t["relative_l2"] for t in items),
                          "max_rmse": max(t["rmse"] for t in items), "max_mae": max(t["mae"] for t in items)}
        require(per_role[role] == result["per_role"][role], "开发按角色总计不一致")
    dev_matching: dict[str, int | float] = {key: sum(r["matching"][key] for r in dev_rows) for key in ("reference_count", "actual_count", "matched")}
    dev_matching["reference_fraction"] = dev_matching["matched"] / dev_matching["reference_count"]
    dev_matching["actual_fraction"] = dev_matching["matched"] / dev_matching["actual_count"]
    require(dev_matching == result["detection_matching"], "开发检测匹配总计不一致")
    require(numeric_images == 40 and result["passed"] is True, "开发结论与重算门槛不符")
    require(audit["independent_acceptance_samples_accessed"] == provenance["development"]["independent_acceptance_samples_accessed"] == 0, "验收数据访问状态发生改变，需重新审核汇报")
    profile_path = g + provenance["cpu_execution"]["final_profile_path"]
    source.sha(profile_path, provenance["cpu_execution"]["final_profile_sha256"])
    placement_rows = tools.parse_perf(source.root / profile_path)
    sigmoid_rows = [r for r in placement_rows if r["type"] == "Sigmoid"]
    require(len(sigmoid_rows) == 3 and all(r["target"] == "CPU" for r in sigmoid_rows), "最终分类Sigmoid CPU归属证据不符")
    crosscheck = source.json(g + "evidence/model/development40/lite_bridge_crosscheck.json")
    require(crosscheck["all_equal"] is True and all(crosscheck["elementwise_equal"]), "Lite2/桥接交叉核对不通过")
    general = {"status": "development_passed_only", "formal_acceptance": False,
               "strategy": provenance["selected_strategy"], "recorded_utc": provenance["recorded_utc"],
               "model_sha256": result["rknn_sha256"], "original_onnx_sha256": result["onnx_sha256"],
               "development_sample_count": len(dev_rows), "development_execution_success_count": len(dev_rows),
               "numeric_passed_images": numeric_images, "numeric_passed_tensors": numeric_tensors,
               "historical_strict_2pct_passed_images": strict_images, "historical_strict_2pct_passed_tensors": strict_tensors,
               "historical_strict_2pct_failed_tensors": sum(len(v) for v in by_role.values()) - strict_tensors,
               "per_role": per_role, "detection_matching": dev_matching,
               "quality_protocol": {"protocol_id": protocol["protocol_id"], "declared_utc": protocol["declared_utc"], "tensor_gates": gates,
                                    "scope": protocol["scope"], "evaluation_minimum": protocol["evaluation"]["minimum"]},
               "cpu_sigmoid_placement": [{k: row[k] for k in ("id", "name", "type", "target", "dtype")} for row in sigmoid_rows],
               "lite_bridge_all_nine_equal": crosscheck["all_equal"], "independent_acceptance_samples_tested": 0,
               "independent_acceptance_expected": protocol["dataset"]["acceptance_count"],
               "independent_acceptance_status": "未正式验收", "independent_acceptance_metrics": None,
               "performance_status": "开发逐图计时包含验证/保存等开销，不是正式性能基准；没有合并进首轮节点图",
               "note": "新的按输出语义门槛通过40图开发检查；旧2%逐张量门槛仍保留失败，不能追认旧模型通过或宣布4960图验收完成。",
               "source": g + "evidence/model/development40/development_results.json"}
    return {"formal_acceptance": False, "home": home, "general": general}


def save_report(output, report):
    """紧凑UTF-8字节直接写入，Windows不转换换行；回执哈希就是落盘字节。"""
    payload = (json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    Path(output).write_bytes(payload)
    return {"output": Path(output).resolve().as_posix(), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "benchmarks/presentation/report_data.json"
    require(output.parent == root / "benchmarks/presentation", "仅允许写入本任务 presentation 目录")
    report = collect(root)
    receipt = save_report(output, report)
    print(json.dumps({**receipt,
                      "validation": report["validation"], "baseline_stages_ms": report["baseline"]["stages_ms"],
                      "detail_total_ms": report["detail"]["node_time_sum_ms"],
                      "representative_image_id": report["hotspots"]["representative"]["image_id"],
                      "general_logged_downloads": report["datasets"]["general"]["download"]["logged_success_count"],
                      "formal_acceptance": report["quality"]["formal_acceptance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
