"""验证汇报证据拒绝伪边，且真实产物可以从原始数据重建。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks/presentation/collect_report_data.py"


def collector():
    assert SCRIPT.is_file(), "尚未实现只读汇报数据生成器"
    spec = importlib.util.spec_from_file_location("presentation_collector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (ROOT / "benchmarks/yolov8n_home/graphs/compiled_graph.json").exists(),
                    reason="此集成检查需要首轮原始图归档；仓库仅保留汇报派生图")
def test_rejects_fabricated_edge_even_when_graph_counts_are_unchanged():
    """节点数量不变、但张量来源被换掉的图必须失败。"""
    module = collector()
    graph = json.loads((ROOT / "benchmarks/yolov8n_home/graphs/compiled_graph.json").read_text(encoding="utf-8"))
    module.validate_graph_integrity(graph)
    broken = copy.deepcopy(graph)
    broken["edges"][0]["source"] = 3
    with pytest.raises(ValueError, match="边|producer|生产者"):
        module.validate_graph_integrity(broken)


@pytest.mark.skipif(not (ROOT / "benchmarks/yolov8n_home/graphs/compiled_graph.json").exists(),
                    reason="此集成检查需要首轮模型、数据与运行归档")
def test_real_report_uses_per_sample_aggregation_and_preserves_failed_quality():
    """真实文件贯穿测试；均值守恒，分位数不得通过节点分位数相加获得。"""
    module = collector()
    assert callable(getattr(module, "collect", None)), "尚未实现真实数据汇总入口"
    report = module.collect(ROOT)
    detail = report["detail"]
    assert len(detail["nodes"]) == 108
    assert len(detail["edges"]) == 138
    assert len(detail["tensors"]) == 257
    assert detail["report_count"] == 40
    assert report["baseline"]["execution_count"] == 400
    assert report["validation"]["status"] == "passed"
    assert report["quality"]["home"]["status"] == "failed"
    assert report["quality"]["formal_acceptance"] is False
    assert sum(n["mean_ms"] for n in detail["nodes"]) == pytest.approx(detail["node_time_sum_ms"]["mean"])
    assert sum(n["p95_ms"] for n in detail["nodes"]) != pytest.approx(detail["node_time_sum_ms"]["p95"])
    modules = {m["id"]: m for m in detail["modules"]}
    assert modules["model.2"]["node_ids"] == [4, 5, 6, 7, 8, 9, 10]
    assert not set(modules["model.2"]["node_ids"]) & set(modules["model.22"]["node_ids"])
    assert modules["input"]["node_ids"] == [1]
    assert len(modules["output"]["node_ids"]) == 9
    for hotspot in report["hotspots"]["representative"]["hotspots"]:
        assert hotspot["replayable"] is False


def test_saved_checksum_covers_exact_utf8_bytes_on_windows(tmp_path):
    """防止 Windows 文本换行转换让宣称的哈希偏离实际文件。"""
    import hashlib
    module = collector()
    assert callable(getattr(module, "save_report", None)), "尚未实现精确字节写入"
    output = tmp_path / "报告.json"
    sample = {"说明": "测试夹具，非实测证据", "节点": [1, 2]}
    receipt = module.save_report(output, sample)
    raw = output.read_bytes()
    assert json.loads(raw) == sample
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["bytes"] == len(raw)
    assert b"\r\n" not in raw


def test_collection_source_uses_only_hash_matching_historical_snapshot(tmp_path):
    """修复当前采集器后，历史结果仍须绑定采集时的原始源码。"""
    import hashlib
    module = collector()
    frozen = b"# synthetic historical source, not run evidence\n"
    digest = hashlib.sha256(frozen).hexdigest()
    current = tmp_path / "benchmarks/yolov8n_home/scripts/run_board.py"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"# synthetic changed source\n")
    relative = f"benchmarks/fixtures/yolov8n_home_sources/{digest}/run_board.py"
    archived = tmp_path / relative
    archived.parent.mkdir(parents=True)
    archived.write_bytes(frozen)
    sources = module.Sources(tmp_path)
    assert sources.collection_script("run_board.py", digest) == relative
    assert sources.entries[relative]["sha256"] == digest
    assert "benchmarks/yolov8n_home/scripts/run_board.py" not in sources.entries
    archived.write_bytes(b"# synthetic corrupted historical source\n")
    with pytest.raises(ValueError, match="SHA-256"):
        module.Sources(tmp_path).collection_script("run_board.py", digest)
    with pytest.raises(ValueError, match="文件名"):
        module.Sources(tmp_path).collection_script("../run_board.py", digest)
