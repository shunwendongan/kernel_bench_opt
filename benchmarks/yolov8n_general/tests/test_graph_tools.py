"""合成夹具单元测试与已有真实 smoke 回归；不执行板端采样。"""

import copy
import importlib.util
import json
from pathlib import Path
import struct

import flatbuffers
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "graph_tools.py"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
HEADER = (
    "ID OpType DataType Target InputShape OutputShape "
    "Cycles(DDR/NPU/Total) Time(us) MacUsage(%) WorkLoad(0/1/2) RW(KB) FullName"
)
# 字段顺序来自交接包 profiling/parse_detail.py；所有值仅用于单元测试。
NPU_ROW = (
    "1 Conv FP16 NPU 1,3,640,640 1,16,320,320 "
    "123456789012/234567890123/345678901234 1250.5 38.25% "
    "33.33%/33.33%/33.34% 128.25 /model.0/conv/Conv with suffix"
)
CPU_ROW = (
    "2 Reshape FP16 CPU 1,16,320,320 1,16,102400 "
    "0/0/0 25.0 0.00 0.00%/0.00%/0.00% 0.00 /model.22/Reshape"
)


def tools():
    assert MODULE.is_file(), "缺少 graph_tools.py 功能实现"
    spec = importlib.util.spec_from_file_location("yolo_graph_tools", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(tmp_path, body):
    path = tmp_path / "synthetic_only.perf.txt"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_perf_handles_cpu_long_cycles_and_fullname(tmp_path):
    rows = tools().parse_perf(report(tmp_path, HEADER + "\n" + NPU_ROW + "\n" + CPU_ROW))
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["type"] == "Conv"
    assert rows[0]["name"] == "/model.0/conv/Conv with suffix"
    assert rows[0]["cycles"] == [123456789012, 234567890123, 345678901234]
    assert rows[0]["time_us"] == 1250.5
    assert rows[0]["workload_pct"] == [33.33, 33.33, 33.34]
    assert rows[0]["rw_kb"] == 128.25
    assert rows[1]["target"] == "CPU"
    assert rows[1]["time_us"] == 25.0


@pytest.mark.parametrize(
    "body",
    [
        HEADER + "\n" + NPU_ROW + "\n" + NPU_ROW,
        HEADER + "\n" + NPU_ROW.replace(" NPU ", " GPU "),
        HEADER + "\n" + NPU_ROW.replace(" 1250.5 ", " nan "),
        HEADER + "\n" + NPU_ROW.replace(" 1250.5 ", " -1 "),
        HEADER + "\n" + NPU_ROW.replace("33.33%/33.33%/33.34%", "33.33/33.33/33.34"),
        HEADER + "\n" + NPU_ROW.rsplit(" 128.25", 1)[0],
        HEADER + "\n" + NPU_ROW + "\n2 broken row",
        HEADER + "\n" + NPU_ROW + "\nx broken row",
        HEADER + "\n" + NPU_ROW + "\n" + HEADER + "\n" + CPU_ROW,
        NPU_ROW,
        HEADER,
    ],
)
def test_parse_perf_fails_closed(tmp_path, body):
    with pytest.raises(ValueError):
        tools().parse_perf(report(tmp_path, body))


def test_parse_perf_real_smoke():
    path = FIXTURES / "yolov8n_home_bridge_smoke.perf.txt"
    rows = tools().parse_perf(path)
    assert len(rows) == 108
    assert [row["id"] for row in rows] == list(range(1, 109))
    assert sum(row["time_us"] for row in rows) == 60431
    assert rows[0]["mac_usage_pct"] is None
    assert rows[4]["mac_usage_pct"] is None  # NPU Split 也未报告 MAC 利用率。
    assert rows[99]["mac_usage_pct"] == [12.31, 12.31, 12.31]
    assert all(row["mac_usage_pct"] is None for row in rows if row["target"] == "CPU")
    assert sum(row["time_us"] for row in rows if row["target"] == "CPU") == 356
    assert sum(row["time_us"] for row in rows if row["target"] == "NPU") == 60075
    # 独立读取原报告 ranking，核对全部类型和 target，避免只检查总数相等。
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.split()[:2] == ["OpType", "CallNumber"])
    ranking = [line.split() for line in lines[start + 1:]
               if line.strip() and not line.startswith("-") and line.split()[0] != "Total"]
    assert len(ranking) == 11
    assert {fields[0] for fields in ranking} == {row["type"] for row in rows}
    for op_type, count, cpu, gpu, npu, total, ratio in ranking:
        selected = [row for row in rows if row["type"] == op_type]
        assert len(selected) == int(count)
        for target, duration in [("CPU", cpu), ("GPU", gpu), ("NPU", npu)]:
            assert sum(row["time_us"] for row in selected if row["target"] == target) == float(duration)
        assert sum(row["time_us"] for row in selected) == float(total)
        assert float(ratio.rstrip("%")) == pytest.approx(float(total) / 60431 * 100, abs=0.005)


@pytest.mark.parametrize("spacing", [" ", "\t", "   \t  "])
def test_parse_perf_real_report_does_not_depend_on_column_width(tmp_path, spacing):
    source = FIXTURES / "yolov8n_home_bridge_smoke.perf.txt"
    body = "\n".join(spacing.join(line.split()) for line in source.read_text(encoding="utf-8").splitlines())
    assert tools().parse_perf(report(tmp_path, "\ufeff" + body + "\x00")) == tools().parse_perf(source)


@pytest.mark.parametrize("mac, expected", [
    ("38.25%", 38.25), ("38.25", 38.25), ("0", 0.0), ("", None),
    ("1/12.31/80", [1.0, 12.31, 80.0]), ("1%/12.31%/80%", [1.0, 12.31, 80.0]),
])
def test_parse_perf_preserves_mac_scalar_triple_or_absence(tmp_path, mac, expected):
    row = NPU_ROW.replace("38.25%", mac)
    parsed = tools().parse_perf(report(tmp_path, HEADER + "\n" + row))[0]
    assert parsed["mac_usage_pct"] == expected
    assert parsed["workload_pct"] == [33.33, 33.33, 33.34]
    assert parsed["rw_kb"] == 128.25
    assert parsed["name"] == "/model.0/conv/Conv with suffix"


@pytest.mark.parametrize("mac", ["nan", "1e999", "101", "1/2", "1/2/3/4", "1/-2/3", "1/101/3"])
def test_parse_perf_rejects_invalid_mac(tmp_path, mac):
    with pytest.raises(ValueError):
        tools().parse_perf(report(tmp_path, HEADER + "\n" + NPU_ROW.replace("38.25%", mac)))


def test_real_smoke_binds_native_model_and_selects_four_connected_hotspots():
    module = tools()
    rows = module.parse_perf(FIXTURES / "yolov8n_home_bridge_smoke.perf.txt")
    native = json.loads((FIXTURES / "yolov8n_home_native_graph.json").read_text(encoding="utf-8"))
    graph = module.bind_graph(native, rows)
    assert (len(graph["nodes"]), len(graph["tensors"]), len(graph["edges"])) == (108, 257, 138)
    assert graph["mapping"] == {
        "method": "index+1/type/FullName", "node_matches": 108,
        "unique_producers": True, "acyclic": True,
    }
    assert graph["time_semantics"] == "sum_node_times_not_wall_clock"
    hotspots = module.select_hotspots(graph, rows, max_hotspots=4)
    assert [(h["name"], h["time_us"], h["node_ids"]) for h in hotspots] == [
        ("model.2", 8396, [4, 5, 6, 7, 8, 9, 10]),
        ("model.4", 7122, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        ("model.15", 5118, [57, 58, 59, 60, 61, 62]),
        ("model.22#2", 4074, [66, 72, 74, 78]),
    ]
    assert [h["output_tensor_ids"] for h in hotspots] == [[154], [166], [213], [225, 228]]
    assert all(h["subgraph_io_available"] is False and h["replayable"] is False for h in hotspots)


@pytest.mark.parametrize("prefix, suffix", [
    ("unrecognized report text\n", ""),
    ("", "\nTotal nonsense: 1275.5"),
    ("", "\nTotal Operator Elapsed Per Frame Time(us): 1"),
])
def test_parse_perf_rejects_unknown_text_and_invalid_totals(tmp_path, prefix, suffix):
    with pytest.raises(ValueError):
        tools().parse_perf(report(tmp_path, prefix + HEADER + "\n" + NPU_ROW + suffix))


@pytest.mark.parametrize("damage", [
    "node", "node_missing", "node_after_totals", "duplicate_report", "duplicate_title",
    "elapsed_total", "memory_total_nan", "missing_memory", "duplicate_memory",
    "ranking_title", "ranking_header", "ranking_row", "ranking_count", "ranking_cpu",
    "ranking_gpu", "ranking_npu", "ranking_time", "ranking_ratio", "ranking_missing_type",
    "ranking_extra_type", "ranking_duplicate_type", "ranking_duplicate_header",
    "ranking_duplicate_table", "ranking_missing_total", "ranking_total_cpu",
    "ranking_total_gpu", "ranking_total_npu", "ranking_total_time", "ranking_duplicate_total",
    "unknown_prefix", "unknown_between", "unknown_suffix",
])
def test_parse_perf_real_report_rejects_corruption(tmp_path, damage):
    source = FIXTURES / "yolov8n_home_bridge_smoke.perf.txt"
    lines = source.read_text(encoding="utf-8").splitlines()
    node = next(i for i, line in enumerate(lines) if line.split()[:1] == ["2"])
    elapsed = next(i for i, line in enumerate(lines) if line.startswith("Total Operator"))
    memory = elapsed + 1
    title = next(i for i, line in enumerate(lines) if "Operator Time Consuming Ranking Table" in line)
    header = next(i for i, line in enumerate(lines) if line.split()[:2] == ["OpType", "CallNumber"])
    ranking = next(i for i, line in enumerate(lines) if line.split()[:1] == ["ConvExSwish"])
    total = next(i for i, line in enumerate(lines) if i > header and line.split()[:1] == ["Total"])
    if damage == "node":
        lines[node] = "2 broken row"
    elif damage == "node_missing":
        del lines[node]
    elif damage == "node_after_totals":
        lines.append(lines[node])
    elif damage == "duplicate_report":
        lines += lines[:]
    elif damage == "duplicate_title":
        lines.insert(2, "Network Layer Information Table")
    elif damage == "elapsed_total":
        lines[elapsed] = "Total Operator Elapsed Per Frame Time(us): 1"
    elif damage == "memory_total_nan":
        lines[memory] = "Total Memory Read/Write Per Frame Size(KB): nan"
    elif damage == "missing_memory":
        del lines[memory]
    elif damage == "duplicate_memory":
        lines.insert(memory, lines[memory])
    elif damage == "ranking_title":
        lines[title] += " unknown"
    elif damage == "ranking_header":
        lines[header] = lines[header].replace("CPUTime(us)", "CPUTime(ms)")
    elif damage == "ranking_row":
        lines[ranking] = "ConvExSwish broken row"
    elif damage in {"ranking_count", "ranking_cpu", "ranking_gpu", "ranking_npu", "ranking_time", "ranking_ratio"}:
        column = {"ranking_count": 1, "ranking_cpu": 2, "ranking_gpu": 3,
                  "ranking_npu": 4, "ranking_time": 5, "ranking_ratio": 6}[damage]
        fields = lines[ranking].split()
        fields[column] = "1%" if column == 6 else str(int(fields[column]) + 1)
        lines[ranking] = " ".join(fields)
    elif damage == "ranking_missing_type":
        del lines[ranking]
    elif damage == "ranking_extra_type":
        lines.insert(ranking, "Unknown 1 0 0 0 0 0.00%")
    elif damage == "ranking_duplicate_type":
        lines.insert(ranking, lines[ranking])
    elif damage == "ranking_duplicate_header":
        lines.insert(ranking, lines[header])
    elif damage == "ranking_duplicate_table":
        lines += lines[title:]
    elif damage == "ranking_missing_total":
        del lines[total:]
    elif damage.startswith("ranking_total_"):
        column = {"ranking_total_cpu": 1, "ranking_total_gpu": 2,
                  "ranking_total_npu": 3, "ranking_total_time": 4}[damage]
        fields = lines[total].split()
        fields[column] = str(int(fields[column]) + 1)
        lines[total] = " ".join(fields)
    elif damage == "ranking_duplicate_total":
        lines.append(lines[total])
    elif damage == "unknown_prefix":
        lines.insert(0, "unknown text")
    elif damage == "unknown_between":
        lines.insert(ranking, "unknown text")
    elif damage == "unknown_suffix":
        lines.append("unknown text")
    with pytest.raises(ValueError):
        tools().parse_perf(report(tmp_path, "\n".join(lines)))


def synthetic_native_bytes():
    """用独立 FlatBuffers Builder 构造合成模型，避免测试复制被测 Reader。"""
    builder = flatbuffers.Builder(1024)

    def vector(values, tables=False):
        builder.StartVector(4, len(values), 4)
        for value in reversed(values):
            if tables:
                builder.PrependUOffsetTRelative(value)
            else:
                builder.PrependInt32(value)
        return builder.EndVector()

    def table(count, fields):
        builder.StartObject(count)
        for slot, value in fields.items():
            builder.PrependUOffsetTRelativeSlot(slot, value, 0)
        return builder.EndObject()

    tensors = []
    for name, kind in [("images", 0), ("weight", 4), ("output", 0)]:
        name_offset = builder.CreateString(name)
        shape_offset = vector([1, 3, 4, 4])
        builder.StartObject(6)
        builder.PrependInt8Slot(0, 1, 0)
        builder.PrependInt8Slot(2, kind, 0)
        builder.PrependUOffsetTRelativeSlot(4, shape_offset, 0)
        builder.PrependUOffsetTRelativeSlot(5, name_offset, 0)
        tensors.append(builder.EndObject())
    node = table(10, {
        1: builder.CreateString("Conv"), 2: builder.CreateString("/model.0/conv/Conv"),
        4: vector([0, 1]), 5: vector([2]),
    })
    graph = table(4, {
        0: vector(tensors, tables=True), 1: vector([node], tables=True),
        2: vector([0]), 3: vector([2]),
    })
    root = table(14, {
        1: builder.CreateString("synthetic-test-only"),
        2: vector([graph], tables=True), 3: builder.CreateString("unit-test-builder"),
    })
    builder.Finish(root, file_identifier=b"RKNN")
    payload = bytes(builder.Output())
    header = bytearray(64)
    header[:8] = b"RKNN\x00\x00\x00\x00"
    struct.pack_into("<Q", header, 16, len(payload))
    outer = json.dumps({"synthetic_unit_test_only": True}).encode("utf-8")
    return bytes(header) + payload + struct.pack("<Q", len(outer)) + outer


def test_extract_graph_reads_native_container(tmp_path):
    module = tools()
    assert hasattr(module, "extract_graph"), "缺少 native 解析功能"
    path = tmp_path / "synthetic_only.rknn"
    path.write_bytes(synthetic_native_bytes())
    native = module.extract_graph(path)
    assert native["metadata"]["format"] == "synthetic-test-only"
    assert native["metadata"]["generator"] == "unit-test-builder"
    assert native["metadata"]["runtime"] is None
    assert native["outer"] == {"synthetic_unit_test_only": True}
    graph = native["graphs"][0]
    assert graph["inputs"] == [0] and graph["outputs"] == [2]
    assert graph["nodes"][0]["inputs"] == [0, 1]
    assert graph["nodes"][0]["name"] == "/model.0/conv/Conv"
    assert graph["nodes"][0]["outputs"] == [2]
    assert graph["tensors"][1]["kind"] == 4
    assert graph["tensors"][2]["shape"] == [1, 3, 4, 4]


@pytest.mark.parametrize("damage", ["empty", "magic", "identifier", "length", "root", "truncated", "json"])
def test_extract_graph_rejects_corrupt_container(tmp_path, damage):
    module = tools()
    assert hasattr(module, "extract_graph"), "缺少 native 解析功能"
    data = bytearray(synthetic_native_bytes())
    if damage == "empty":
        data = b""
    elif damage == "magic":
        data[0] = 0
    elif damage == "identifier":
        data[68:72] = b"FAIL"
    elif damage == "length":
        struct.pack_into("<Q", data, 16, len(data) + 1)
    elif damage == "root":
        struct.pack_into("<I", data, 64, len(data) + 1)
    elif damage == "truncated":
        data = data[:-1]
    elif damage == "json":
        data[-1] = ord("!")
    path = tmp_path / "broken_synthetic.rknn"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        module.extract_graph(path)


def small_native():
    """合成依赖：InputOperator → Conv → Add；张量 1 没有 producer。"""
    return {"metadata": {}, "outer": {}, "graphs": [{
        "nodes": [
            {"index": 0, "type": "InputOperator", "name": "InputOperator:images", "inputs": [], "outputs": [0]},
            {"index": 1, "type": "Conv", "name": "/model.0/conv/Conv", "inputs": [0, 1], "outputs": [2]},
            {"index": 2, "type": "Add", "name": "/model.1/Add", "inputs": [2, 2, 1], "outputs": [3]},
        ],
        "tensors": [
            {"index": i, "name": f"tensor_{i}", "shape": [1], "data_type": 1, "kind": 7 if i == 1 else 0}
            for i in range(4)
        ],
        "inputs": [0], "outputs": [3],
    }]}


def synthetic_rows(native, times=None):
    """人工耗时只用于单元测试，不输出为测量证据。"""
    return [{
        "id": n["index"] + 1, "type": n["type"], "name": n["name"],
        "target": "CPU" if n["type"] == "InputOperator" else "NPU",
        "time_us": times[i] if times is not None else 10.0,
    } for i, n in enumerate(native["graphs"][0]["nodes"])]


def test_bind_graph_matches_ids_and_retains_nonproduced_tensors():
    module = tools()
    assert hasattr(module, "bind_graph"), "缺少严格图映射功能"
    native = small_native()
    rows = synthetic_rows(native)
    original = copy.deepcopy((native, rows))
    graph = module.bind_graph(native, list(reversed(rows)))
    assert [n["id"] for n in graph["nodes"]] == [1, 2, 3]
    assert graph["tensors"][1]["producer"] is None
    assert graph["tensors"][1]["kind"] == 7
    assert graph["tensors"][1]["consumers"] == [2, 3]
    assert graph["tensors"][0]["producer"] == 1
    assert graph["inputs"] == [0] and graph["outputs"] == [3]
    assert [(e["source"], e["target"], e["tensor_id"], e["input_slot"]) for e in graph["edges"]] == [
        (1, 2, 0, 0), (2, 3, 2, 0), (2, 3, 2, 1),
    ]
    assert graph["topological_order"] == [1, 2, 3]
    assert (native, rows) == original


def test_bind_graph_accepts_dag_with_non_topological_indices():
    module = tools()
    assert hasattr(module, "bind_graph"), "缺少严格图映射功能"
    native = small_native()
    nodes = native["graphs"][0]["nodes"]
    nodes[1], nodes[2] = nodes[2], nodes[1]
    for index, node in enumerate(nodes):
        node["index"] = index
    graph = module.bind_graph(native, synthetic_rows(native))
    assert graph["topological_order"] == [1, 3, 2]


@pytest.mark.parametrize("damage", [
    "multi_graph", "node_index", "tensor_index", "duplicate_producer", "cycle", "self_cycle",
    "negative_tensor", "unknown_tensor", "bad_output", "missing_rows", "extra_row",
    "duplicate_id", "wrong_id", "wrong_type", "wrong_name", "negative_time", "nan_time", "bad_target",
])
def test_bind_graph_rejects_invalid_mapping(damage):
    module = tools()
    assert hasattr(module, "bind_graph"), "缺少严格图映射功能"
    native = small_native()
    rows = synthetic_rows(native)
    graph = native["graphs"][0]
    nodes = graph["nodes"]
    if damage == "multi_graph":
        native["graphs"].append(copy.deepcopy(graph))
    elif damage == "node_index":
        nodes[1]["index"] = 0
    elif damage == "tensor_index":
        graph["tensors"][1]["index"] = 0
    elif damage == "duplicate_producer":
        nodes[2]["outputs"].append(2)
    elif damage == "cycle":
        nodes[1]["inputs"] = [3, 1]
    elif damage == "self_cycle":
        nodes[1]["inputs"] = [2, 1]
    elif damage == "negative_tensor":
        nodes[1]["inputs"].append(-1)
    elif damage == "unknown_tensor":
        nodes[1]["outputs"].append(500)
    elif damage == "bad_output":
        graph["outputs"].append(99)
    elif damage == "missing_rows":
        rows.pop()
    elif damage == "extra_row":
        rows.append(dict(rows[-1], id=4))
    elif damage == "duplicate_id":
        rows[1]["id"] = 1
    elif damage == "wrong_id":
        rows[1]["id"] = 9
    elif damage == "wrong_type":
        rows[1]["type"] = "ConvExSwish"
    elif damage == "wrong_name":
        rows[1]["name"] += "different"
    elif damage == "negative_time":
        rows[1]["time_us"] = -1
    elif damage == "nan_time":
        rows[1]["time_us"] = float("nan")
    elif damage == "bad_target":
        rows[1]["target"] = "GPU"
    with pytest.raises(ValueError):
        module.bind_graph(native, rows)


def hotspot_native():
    native = small_native()
    graph = native["graphs"][0]
    graph["nodes"][2]["name"] = "/model.0/Add"
    specifications = [
        ("/model.0/separate/Conv", [0, 1], [4]),
        ("/model.1/Add", [3, 4], [5]),
        ("unmapped_bridge", [5], [6]),
        ("Conv:model.22.cv2.0", [6], [7]),
        ("/model/22/final", [7], [8]),
        ("unmapped_tail", [8], [9]),
    ]
    for name, inputs, outputs in specifications:
        graph["nodes"].append({"index": len(graph["nodes"]), "type": "Conv", "name": name,
                               "inputs": inputs, "outputs": outputs})
    for index in range(4, 10):
        graph["tensors"].append({"index": index, "name": f"tensor_{index}", "shape": [1],
                                 "data_type": 1, "kind": 0})
    # 张量 2 也声明为图输出，即使它在模块内部仍有消费者。
    graph["outputs"] = [2, 9]
    return native


def test_hotspots_rank_by_sample_sum_and_preserve_true_boundaries():
    module = tools()
    assert hasattr(module, "select_hotspots"), "缺少热点选择功能"
    native = hotspot_native()
    graph = module.bind_graph(native, synthetic_rows(native))
    rows = synthetic_rows(native, [1, 600, 400, 800, 700, 650, 100, 100, 5])
    before = copy.deepcopy((graph, rows))
    hotspots = module.select_hotspots(graph, rows)
    assert [h["node_ids"] for h in hotspots] == [[2, 3], [4], [5], [6]]
    assert [h["rank"] for h in hotspots] == [1, 2, 3, 4]
    assert [h["time_ms"] for h in hotspots] == [1.0, 0.8, 0.7, 0.65]
    assert hotspots[0]["input_tensor_ids"] == [0, 1]
    assert hotspots[0]["output_tensor_ids"] == [2, 3]
    assert hotspots[0]["input_tensors"][1]["kind"] == 7
    assert hotspots[0]["input_tensors"][1]["producer"] is None
    assert all(h["subgraph_io_available"] is False and h["replayable"] is False for h in hotspots)
    assert all(h["time_semantics"] == "sum_node_times_not_wall_clock" for h in hotspots)
    selected = [node for h in hotspots for node in h["node_ids"]]
    assert len(selected) == len(set(selected))
    assert (graph, rows) == before


def test_hotspot_candidates_partition_nodes_into_connected_components():
    module = tools()
    assert hasattr(module, "select_hotspots"), "缺少热点选择功能"
    native = hotspot_native()
    graph = module.bind_graph(native, synthetic_rows(native))
    discovered = set()
    for index in range(len(graph["nodes"])):
        times = [0] * len(graph["nodes"])
        times[index] = 1000
        hotspot = module.select_hotspots(graph, synthetic_rows(native, times), max_hotspots=1)[0]
        discovered.add(tuple(hotspot["node_ids"]))
        members = set(hotspot["node_ids"])
        seen = {min(members)}
        while True:
            expanded = seen | {endpoint for edge in graph["edges"]
                               if edge["source"] in members and edge["target"] in members
                               and (edge["source"] in seen or edge["target"] in seen)
                               for endpoint in (edge["source"], edge["target"])}
            if expanded == seen:
                break
            seen = expanded
        assert seen == members
    assert discovered == {(1,), (2, 3), (4,), (5,), (6,), (7, 8), (9,)}
    flattened = [node for component in discovered for node in component]
    assert sorted(flattened) == list(range(1, len(graph["nodes"]) + 1))


def test_hotspots_use_weak_connectivity_and_keep_unmatched_nodes_independent():
    module = tools()
    assert hasattr(module, "select_hotspots"), "缺少热点选择功能"
    native = hotspot_native()
    native["graphs"][0]["nodes"][3]["inputs"] = [2, 1]
    rows = synthetic_rows(native, [0, 300, 300, 300, 0, 0, 0, 0, 0])
    graph = module.bind_graph(native, rows)
    assert module.select_hotspots(graph, rows, 1)[0]["node_ids"] == [2, 3, 4]
    native = small_native()
    for node in native["graphs"][0]["nodes"]:
        node["name"] = "no_module_match"
    rows = synthetic_rows(native)
    graph = module.bind_graph(native, rows)
    assert [h["node_ids"] for h in module.select_hotspots(graph, rows)] == [[1], [2], [3]]


@pytest.mark.parametrize("limit", [-1, 5, True, 1.5, "4"])
def test_hotspots_reject_invalid_limit(limit):
    module = tools()
    assert hasattr(module, "select_hotspots"), "缺少热点选择功能"
    native = small_native()
    rows = synthetic_rows(native)
    with pytest.raises(ValueError):
        module.select_hotspots(module.bind_graph(native, rows), rows, limit)


def test_hotspots_validate_each_sample_even_for_zero_limit():
    module = tools()
    assert hasattr(module, "select_hotspots"), "缺少热点选择功能"
    native = small_native()
    rows = synthetic_rows(native)
    graph = module.bind_graph(native, rows)
    assert module.select_hotspots(graph, rows, 0) == []
    rows[0]["name"] = "another_model"
    with pytest.raises(ValueError):
        module.select_hotspots(graph, rows, 0)
