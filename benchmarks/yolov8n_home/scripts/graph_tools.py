"""RKNN native 图、单次 PERF_DETAIL 与 YOLO 热点的只读处理。

不调用 Runtime、不采集性能、不推断未捕获的边界张量值。异常输入一律
抛出 ValueError；文件访问错误保留为 OSError。节点时间只表示节点耗时求和。
"""

from __future__ import annotations

import copy
from decimal import Decimal
import heapq
import json
import math
import mmap
from pathlib import Path
import re
import struct


_PERF_HEADER = (
    "ID OpType DataType Target InputShape OutputShape "
    "Cycles(DDR/NPU/Total) Time(us) MacUsage(%) WorkLoad(0/1/2) RW(KB) FullName"
).split()
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PERF_ROW = re.compile(
    rf"(?P<id>[1-9]\d*)\s+(?P<type>\S+)\s+(?P<dtype>\S+)\s+"
    rf"(?P<target>CPU|NPU)\s+(?P<input_shape>\S+)\s+(?P<output_shape>\S+)\s+"
    rf"(?P<cycles>\d+/\d+/\d+)\s+(?P<time_us>{_NUMBER})\s+"
    rf"(?:(?P<mac_usage>{_NUMBER}%?(?:/{_NUMBER}%?/{_NUMBER}%?)?)\s+)?"
    rf"(?P<workload>{_NUMBER}%/{_NUMBER}%/{_NUMBER}%)\s+"
    rf"(?P<rw_kb>{_NUMBER})\s+(?P<name>\S(?:.*\S)?)"
)
_TIME_TOTAL = re.compile(rf"Total Operator Elapsed Per Frame Time\(us\): ({_NUMBER})")
_MEMORY_TOTAL = re.compile(rf"Total Memory Read/Write Per Frame Size\(KB\): ({_NUMBER})")
_RANKING_TITLE = "Operator Time Consuming Ranking Table"
_RANKING_HEADER = "OpType CallNumber CPUTime(us) GPUTime(us) NPUTime(us) TotalTime(us) TimeRatio(%)".split()
_RANKING_ROW = re.compile(
    rf"(\S+)\s+([1-9]\d*)\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})%"
)
_RANKING_TOTAL = re.compile(rf"Total\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})")


def _finite_nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: 必须是非负有限数值")
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label}: 必须是非负有限数值")
    return value


def _perf_times(rows):
    try:
        return [math.fsum(row["time_us"] for row in rows if row["target"] == target)
                for target in ("CPU", "GPU", "NPU")] + [math.fsum(row["time_us"] for row in rows)]
    except OverflowError as exc:
        raise ValueError("PERF_DETAIL 节点耗时求和溢出") from exc


def _validate_perf_tail(lines, rows, elapsed, label):
    """严格识别尾表，并用节点值核验所有可比的耗时、调用数和占比。"""
    actual = _perf_times(rows)
    elapsed = _finite_nonnegative(float(elapsed), label)
    if not math.isclose(actual[-1], elapsed, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{label}: Total Operator 耗时与节点求和不一致")
    tail = list(lines)
    if not tail:
        raise ValueError(f"{label}: 缺少 Total Memory")
    label, line = tail[0]
    memory = _MEMORY_TOTAL.fullmatch(" ".join(line.split()))
    if memory is None:
        raise ValueError(f"{label}: 缺少有效 Total Memory")
    # 总计有小数，而节点 RW 可能仅显示整数；不能将显示值之和当作精确总计。
    _finite_nonnegative(float(memory[1]), label)
    if len(tail) == 1:
        return
    if (len(tail) < 5 or " ".join(tail[1][1].split()) != _RANKING_TITLE
            or tail[2][1].split() != _RANKING_HEADER):
        raise ValueError(f"{tail[1][0]}: 排名表标题/表头无效或尾表不完整")
    groups = {}
    for row in rows:
        groups.setdefault(row["type"], []).append(row)
    seen = set()
    for label, line in tail[3:-1]:
        match = _RANKING_ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"{label}: 无法完整解析排名行")
        op_type, calls, *values, ratio_text = match.groups()
        if op_type in seen or op_type not in groups:
            raise ValueError(f"{label}: 排名 OpType 重复或不在节点表内")
        times = [_finite_nonnegative(float(value), label) for value in values]
        expected = _perf_times(groups[op_type])
        if int(calls) != len(groups[op_type]) or any(
                not math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(times, expected)):
            raise ValueError(f"{label}: 排名调用数/target 耗时与节点聚合不一致")
        ratio = _finite_nonnegative(float(ratio_text), label)
        expected_ratio = expected[-1] / actual[-1] * 100 if actual[-1] else 0.0
        # 占比按报告显示的小数精度四舍五入，不要求舍入后的各行恰好合计 100%。
        tolerance = 0.5 * 10.0 ** min(0, int(Decimal(ratio_text).as_tuple().exponent))
        if ratio > 100 or abs(ratio - expected_ratio) > tolerance + 1e-9:
            raise ValueError(f"{label}: 排名 TimeRatio 与节点聚合不一致")
        seen.add(op_type)
    if seen != set(groups):
        raise ValueError(f"{label}: 排名表缺少节点 OpType")
    label, line = tail[-1]
    total = _RANKING_TOTAL.fullmatch(line)
    if total is None:
        raise ValueError(f"{label}: 排名 Total 缺失或无效")
    totals = [_finite_nonnegative(float(value), label) for value in total.groups()]
    if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(totals, actual)):
        raise ValueError(f"{label}: 排名 Total 与节点 target 聚合不一致")


def parse_perf(path: str | Path) -> list[dict]:
    """读取一个样本的一张 PERF_DETAIL 表，保留尾部完整名称。

    按字段语法解析，完全不依赖列宽。CPU/NPU 均保留；所有数据行必须解析，
    ID 不得重复。mac_usage_pct 为标量 float、按原顺序保留的三值 list，或
    未报告时的 None；三值不取均值，也不猜测各分量含义。其他字段保持原义。
    可接受无尾表的节点表；出现 Total 后必须有 elapsed/memory 两项汇总。
    排名表一旦出现必须完整，其调用数、target/总耗时和舍入占比均核对节点值。
    RW 总计仅验证非负有限数值，不与精度较低的节点显示值强行等同。
    仅允许已知标题、分隔线、末尾 C 字符串 NUL；未知文本、损坏行、重复表拒绝。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig").rstrip("\x00")
    if "\x00" in text:
        raise ValueError(f"{path}: 报告中间含 NUL")
    rows, ids = [], set()
    header_seen = False
    title_seen = False
    lines = iter((f"{path}:{number}", raw.strip()) for number, raw in enumerate(text.splitlines(), 1)
                 if raw.strip() and not re.fullmatch(r"[-=+_*\s]+", raw))
    for label, line in lines:
        if line.split() == _PERF_HEADER:
            if header_seen:
                raise ValueError(f"{label}: 重复表头，只接受单次采集")
            header_seen = True
            continue
        if not header_seen:
            if " ".join(line.split()) == "Network Layer Information Table" and not title_seen:
                title_seen = True
                continue
            raise ValueError(f"{label}: 表头之前出现未知文本或重复标题")
        total = _TIME_TOTAL.fullmatch(" ".join(line.split()))
        if total is not None:
            if not rows:
                raise ValueError(f"{label}: 汇总之前缺少节点行")
            _validate_perf_tail(lines, rows, total[1], label)
            break
        match = _PERF_ROW.fullmatch(line)
        if match is None:
            raise ValueError(f"{label}: 无法完整解析 PERF_DETAIL 数据行")
        fields = match.groupdict()
        ident = int(fields["id"])
        if ident in ids:
            raise ValueError(f"{label}: 重复 ID {ident}")
        time_us = _finite_nonnegative(float(fields["time_us"]), label)
        mac_values = [] if fields["mac_usage"] is None else [
            _finite_nonnegative(float(v.rstrip("%")), label)
            for v in fields["mac_usage"].split("/")
        ]
        mac_usage = None if not mac_values else mac_values[0] if len(mac_values) == 1 else mac_values
        workload = [float(v[:-1]) for v in fields["workload"].split("/")]
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in mac_values + workload):
            raise ValueError(f"{label}: 百分比超出 [0, 100]")
        rw_kb = _finite_nonnegative(float(fields["rw_kb"]), label)
        rows.append({
            "id": ident, "type": fields["type"], "name": fields["name"],
            "target": fields["target"], "dtype": fields["dtype"],
            "input_shape": fields["input_shape"], "output_shape": fields["output_shape"],
            "cycles": [int(v) for v in fields["cycles"].split("/")],
            "time_us": time_us, "time_ms": time_us / 1000,
            "mac_usage_pct": mac_usage, "workload_pct": workload,
            "rw_kb": rw_kb, "line_number": int(label.rsplit(":", 1)[1]),
        })
        ids.add(ident)
    if not header_seen or not rows:
        raise ValueError(f"{path}: 缺少有效表头或节点行")
    return rows


class _Reader:
    """只读取公开 RKNN FlatBuffer 字段，所有偏移均限制在声明的 payload 内。"""

    def __init__(self, data, start, length):
        self.data, self.start, self.length = data, start, length
        if start < 0 or length < 8 or start + length > len(data):
            raise ValueError("RKNN payload 越界或过短")

    def bounds(self, pos, size):
        if pos < 0 or size < 0 or pos + size > self.length:
            raise ValueError(f"RKNN FlatBuffer 越界: offset={pos}, size={size}")

    def val(self, pos, fmt):
        self.bounds(pos, struct.calcsize(fmt))
        return struct.unpack_from(fmt, self.data, self.start + pos)[0]

    def field(self, pos, slot, width=4):
        vtable = pos - self.val(pos, "<i")
        vsize = self.val(vtable, "<H")
        object_size = self.val(vtable + 2, "<H")
        if vsize < 4 or vsize % 2 or object_size < 4:
            raise ValueError("RKNN FlatBuffer 表大小无效")
        self.bounds(vtable, vsize)
        self.bounds(pos, object_size)
        offset = self.val(vtable + slot, "<H") if slot < vsize else 0
        if not offset:
            return None
        if offset < 4 or offset + width > object_size:
            raise ValueError("RKNN FlatBuffer 字段越出对象")
        return pos + offset

    def indirect(self, pos):
        offset = self.val(pos, "<I")
        if offset < 4:
            raise ValueError("RKNN FlatBuffer 相对指针无效")
        target = pos + offset
        self.bounds(target, 4)
        return target

    def text(self, pos, slot):
        field = self.field(pos, slot)
        if field is None:
            return None
        target = self.indirect(field)
        length = self.val(target, "<I")
        self.bounds(target + 4, length + 1)
        if self.val(target + 4 + length, "<B") != 0:
            raise ValueError("RKNN FlatBuffer 字符串未终止")
        start = self.start + target + 4
        return self.data[start:start + length].decode("utf-8")

    def vector(self, pos, slot, tables=False):
        field = self.field(pos, slot)
        if field is None:
            return []
        target = self.indirect(field)
        length = self.val(target, "<I")
        self.bounds(target + 4, length * 4)
        return [
            self.indirect(target + 4 + i * 4) if tables
            else self.val(target + 4 + i * 4, "<i")
            for i in range(length)
        ]

    def scalar(self, pos, slot, fmt="<i", default=0):
        field = self.field(pos, slot, struct.calcsize(fmt))
        return default if field is None else self.val(field, fmt)


def extract_graph(model_path: str | Path) -> dict:
    """提取实际 RKNN 的 metadata/graphs/outer，不加载模型或读取可执行代码。

    字段布局沿用交接包 inspect_native.py（Netron rknn-schema.js）。不写死
    模型哈希、节点数、输出形状或数据类型；未知/损坏的容器布局直接报错。
    """
    path = Path(model_path)
    with path.open("rb") as file:
        if file.seek(0, 2) < 80:
            raise ValueError(f"{path}: RKNN 文件过短")
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data[:8] != b"RKNN\x00\x00\x00\x00":
                raise ValueError(f"{path}: RKNN 容器 magic 不匹配")
            length = struct.unpack_from("<Q", data, 16)[0]
            reader = _Reader(data, 64, length)
            if data[68:72] != b"RKNN":
                raise ValueError(f"{path}: RKNN FlatBuffer identifier 不匹配")
            root = reader.indirect(0)
            metadata = {
                key: reader.text(root, slot) for key, slot in (
                    ("format", 6), ("generator", 10), ("compiler", 18),
                    ("runtime", 20), ("source", 22), ("input_json", 28), ("output_json", 30),
                )
            }
            graphs = []
            for graph in reader.vector(root, 8, tables=True):
                nodes = [{
                    "index": i, "type": reader.text(node, 6), "name": reader.text(node, 8),
                    "inputs": reader.vector(node, 12), "outputs": reader.vector(node, 14),
                    "var1": reader.scalar(node, 4), "var2": reader.scalar(node, 10, "<b"),
                    "var4": reader.scalar(node, 18, "<b"), "var5": reader.scalar(node, 20),
                    "var6": reader.scalar(node, 22),
                } for i, node in enumerate(reader.vector(graph, 6, tables=True))]
                tensors = [{
                    "index": i, "name": reader.text(tensor, 14),
                    "shape": reader.vector(tensor, 12),
                    "data_type": reader.scalar(tensor, 4, "<b"),
                    "kind": reader.scalar(tensor, 8, "<b"),
                } for i, tensor in enumerate(reader.vector(graph, 4, tables=True))]
                graphs.append({
                    "nodes": nodes, "tensors": tensors,
                    "inputs": reader.vector(graph, 8), "outputs": reader.vector(graph, 10),
                })
            outer_start = 64 + length
            if outer_start + 8 > len(data):
                raise ValueError(f"{path}: 缺少 outer JSON 长度")
            outer_length = struct.unpack_from("<Q", data, outer_start)[0]
            if outer_start + 8 + outer_length > len(data):
                raise ValueError(f"{path}: outer JSON 越界")
            outer = json.loads(data[outer_start + 8:outer_start + 8 + outer_length])
            if not isinstance(outer, dict):
                raise ValueError(f"{path}: outer JSON 必须是对象")
    return {"metadata": metadata, "graphs": graphs, "outer": outer}


def _indexed_records(records, label):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label}: 必须是非空列表")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or type(record.get("index")) is not int or record["index"] != index:
            raise ValueError(f"{label}[{index}]: index 必须与 native 向量位置相同")
    return copy.deepcopy(records)


def _tensor_refs(values, tensor_count, label, unique=False):
    if not isinstance(values, list) or any(type(t) is not int or not 0 <= t < tensor_count for t in values):
        raise ValueError(f"{label}: 张量索引无效或越界")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{label}: 张量索引重复")
    return list(values)


def _performance_by_id(rows, nodes):
    if not isinstance(rows, list) or len(rows) != len(nodes):
        raise ValueError("PERF_DETAIL 与 native 节点数不一致")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0:
            raise ValueError("PERF_DETAIL ID 必须是正整数")
        ident = row["id"]
        if ident in by_id:
            raise ValueError(f"PERF_DETAIL 重复 ID {ident}")
        if row.get("target") not in ("CPU", "NPU"):
            raise ValueError(f"PERF_DETAIL ID {ident}: Target 必须是 CPU/NPU")
        _finite_nonnegative(row.get("time_us"), f"PERF_DETAIL ID {ident} time_us")
        by_id[ident] = row
    if set(by_id) != set(range(1, len(nodes) + 1)):
        raise ValueError("PERF_DETAIL ID 集合不等于 native index + 1")
    for node in nodes:
        ident = node["index"] + 1
        row = by_id[ident]
        if (node.get("type"), node.get("name")) != (row.get("type"), row.get("name")):
            raise ValueError(f"ID {ident}: native type/FullName 与 PERF_DETAIL 不一致")
    return by_id


def bind_graph(native: dict, rows: list[dict]) -> dict:
    """将一个样本的耗时按 index+1、type、FullName 严格绑定到唯一 native 图。

    返回 nodes/tensors/edges/inputs/outputs。节点 ID 从 1 开始，张量 ID 是原始
    0 起始 index；所有 inputs/outputs 均为张量 ID。tensors 保留原始 kind 并附加
    producer（无 producer 时为 None）及消费者 ID。重复使用同一输入保留 input_slot。
    通过 Kahn 算法验证 DAG，不要求节点编号本身已经按拓扑顺序排列。
    """
    graphs = native.get("graphs") if isinstance(native, dict) else None
    if not isinstance(graphs, list) or len(graphs) != 1 or not isinstance(graphs[0], dict):
        raise ValueError("native 必须恰好包含一张图")
    raw = graphs[0]
    nodes = _indexed_records(raw.get("nodes"), "nodes")
    tensors = _indexed_records(raw.get("tensors"), "tensors")
    for tensor in tensors:
        if type(tensor.get("kind")) is not int:
            raise ValueError(f"tensor {tensor['index']}: 缺少原始 kind")
        if not isinstance(tensor.get("shape"), list) or any(type(v) is not int for v in tensor["shape"]):
            raise ValueError(f"tensor {tensor['index']}: shape 无效")
    count = len(tensors)
    inputs = _tensor_refs(raw.get("inputs"), count, "graph inputs", unique=True)
    outputs = _tensor_refs(raw.get("outputs"), count, "graph outputs", unique=True)
    producers = {}
    for node in nodes:
        ident = node["index"] + 1
        if any(not isinstance(node.get(k), str) or not node[k].strip() for k in ("type", "name")):
            raise ValueError(f"node {ident}: 缺少 type/FullName")
        node["inputs"] = _tensor_refs(node.get("inputs"), count, f"node {ident} inputs")
        node["outputs"] = _tensor_refs(node.get("outputs"), count, f"node {ident} outputs")
        for tensor_id in node["outputs"]:
            if tensor_id in producers:
                raise ValueError(f"tensor {tensor_id}: producer 不唯一")
            producers[tensor_id] = ident
    by_id = _performance_by_id(rows, nodes)
    parents = {i: set() for i in by_id}
    children = {i: set() for i in by_id}
    consumers = {t: set() for t in range(count)}
    edges = []
    for node in nodes:
        ident = node["index"] + 1
        perf = by_id[ident]
        node.update(id=ident, target=perf["target"], time_us=perf["time_us"],
                    time_ms=perf["time_us"] / 1000, performance=copy.deepcopy(perf))
        for slot, tensor_id in enumerate(node["inputs"]):
            consumers[tensor_id].add(ident)
            source = producers.get(tensor_id)
            if source is not None:
                edges.append({"source": source, "target": ident, "tensor_id": tensor_id, "input_slot": slot})
                parents[ident].add(source)
                children[source].add(ident)
    indegree = {ident: len(values) for ident, values in parents.items()}
    ready = [ident for ident, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        ident = heapq.heappop(ready)
        order.append(ident)
        for child in sorted(children[ident]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        raise ValueError("native 依赖图存在有向环")
    input_set, output_set = set(inputs), set(outputs)
    for tensor in tensors:
        index = tensor["index"]
        tensor.update(id=index, producer=producers.get(index), consumers=sorted(consumers[index]),
                      is_graph_input=index in input_set, is_graph_output=index in output_set)
    return {
        "metadata": copy.deepcopy(native.get("metadata", {})),
        "outer": copy.deepcopy(native.get("outer", {})),
        "nodes": nodes, "tensors": tensors, "edges": edges,
        "inputs": inputs, "outputs": outputs, "topological_order": order,
        "mapping": {"method": "index+1/type/FullName", "node_matches": len(nodes),
                    "unique_producers": True, "acyclic": True},
        "time_semantics": "sum_node_times_not_wall_clock",
    }


_YOLO_MODULE = re.compile(r"(?<![\w.])model[./]([0-9]+)(?=[./]|$)")


def select_hotspots(graph: dict, rows: list[dict], max_hotspots: int = 4) -> list[dict]:
    """按当前样本节点耗时选出最多四个连通、互不重叠的热点。

    先按 /model.N/、model.N.xxx 或 /model/N/ 分组，再在每组诱导子图上求
    无向连通分量。共享外部张量不算组内边；无匹配或跨多个模块的融合名称
    单独保留。排序为节点耗时和降序，同耗时按最小节点 ID；不跨样本取均值。
    边界输入包括全部无 producer 张量，kind 原样保留，不猜测常量/权重类型。
    尚未捕获边界值，所以 subgraph_io_available/replayable 固定为 False。
    """
    if type(max_hotspots) is not int or not 0 <= max_hotspots <= 4:
        raise ValueError("max_hotspots 必须是 0 到 4 的整数")
    # 重新核对本样本的身份/时间及实际张量依赖，不能沿用上一样本的耗时。
    bound = bind_graph({"graphs": [graph]}, rows)
    if max_hotspots == 0:
        return []
    nodes = {node["id"]: node for node in bound["nodes"]}
    tensors = bound["tensors"]
    groups, node_group = {}, {}
    for ident, node in nodes.items():
        modules = set(_YOLO_MODULE.findall(node["name"]))
        group = "model." + next(iter(modules)) if len(modules) == 1 else f"unmatched.{ident}"
        groups.setdefault(group, set()).add(ident)
        node_group[ident] = group
    adjacency = {ident: set() for ident in nodes}
    for edge in bound["edges"]:
        source, target = edge["source"], edge["target"]
        if node_group[source] == node_group[target]:
            adjacency[source].add(target)
            adjacency[target].add(source)
    candidates = []
    for group, ids in groups.items():
        remaining = set(ids)
        components = []
        while remaining:
            start = min(remaining)
            component, pending = set(), [start]
            remaining.remove(start)
            while pending:
                ident = pending.pop()
                component.add(ident)
                for neighbor in sorted(adjacency[ident] & remaining):
                    remaining.remove(neighbor)
                    pending.append(neighbor)
            components.append(component)
        for number, component in enumerate(components, 1):
            input_ids = sorted({
                tensor_id for ident in component for tensor_id in nodes[ident]["inputs"]
                if tensors[tensor_id]["producer"] not in component
            })
            output_ids = sorted({
                tensor_id for ident in component for tensor_id in nodes[ident]["outputs"]
                if tensors[tensor_id]["is_graph_output"]
                or any(consumer not in component for consumer in tensors[tensor_id]["consumers"])
            })
            try:
                time_us = math.fsum(nodes[ident]["time_us"] for ident in sorted(component))
            except OverflowError as exc:
                raise ValueError("热点节点耗时求和溢出") from exc
            name = f"{group}#{number}" if len(components) > 1 else group
            candidates.append({
                "name": name, "node_ids": sorted(component),
                "input_tensor_ids": input_ids, "output_tensor_ids": output_ids,
                "input_tensors": [copy.deepcopy(tensors[t]) for t in input_ids],
                "output_tensors": [copy.deepcopy(tensors[t]) for t in output_ids],
                "time_us": time_us, "time_ms": time_us / 1000,
                "time_semantics": "sum_node_times_not_wall_clock",
                "subgraph_io_available": False, "replayable": False,
            })
    candidates.sort(key=lambda item: (-item["time_us"], item["node_ids"][0]))
    return [dict(item, rank=rank) for rank, item in enumerate(candidates[:max_hotspots], 1)]
