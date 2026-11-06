"""不依赖大模型或数据集的离线交付检查。"""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "benchmarks/presentation"
REPORT = ROOT / "汇报材料/2026-11-06_YOLOv8n_RK3588_小组汇报.html"


class Artifact(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.data = []
        self.assets = []
        self.slide_count = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "reportData":
            self.in_data = True
        if tag in ("img", "script", "iframe") and "src" in attrs:
            self.assets.append(attrs["src"])
        if tag == "link" and "href" in attrs:
            self.assets.append(attrs["href"])
        if tag == "section" and "slide" in (attrs.get("class") or "").split():
            self.slide_count += 1

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.data.append(data)


def test_offline_report_preserves_data_identity_and_failed_quality():
    parser = Artifact()
    html = REPORT.read_text(encoding="utf-8")
    parser.feed(html)
    assert parser.slide_count == 13
    assert len(parser.assets) == 2
    assert all(asset.startswith("data:image/") for asset in parser.assets)
    assert "__REPORT_JSON__" not in html and "__INPUT_IMAGE__" not in html
    embedded = json.loads("".join(parser.data))
    raw = (HERE / "report_data.json").read_bytes()
    source = json.loads(raw)
    assert embedded["meta"]["presentation"]["data_sha256"] == hashlib.sha256(raw).hexdigest()
    assert embedded["detail"] == source["detail"]
    assert embedded["quality"]["home"]["status"] == "failed"
    assert embedded["quality"]["general"]["formal_acceptance"] is False
    assert embedded["baseline"]["execution_count"] == 400
    assert embedded["detail"]["report_count"] == 40
    assert embedded["meta"]["presentation"]["template_sha256"] == hashlib.sha256((HERE / "yolo_report.template.html").read_bytes()).hexdigest()


def test_module_edges_are_exactly_the_folded_tensor_dependencies():
    source = json.loads((HERE / "report_data.json").read_text(encoding="utf-8"))
    detail = source["detail"]
    membership = {node["id"]: node["module"] for node in detail["nodes"]}
    assert len(membership) == 108
    folded = {}
    for edge in detail["edges"]:
        pair = membership[edge["source"]], membership[edge["target"]]
        if pair[0] != pair[1]:
            folded.setdefault(pair, set()).add(edge["tensor_id"])
    assert folded == {(edge["source"], edge["target"]): set(edge["tensor_ids"])
                      for edge in detail["module_edges"]}
    all_nodes = [node for module in detail["modules"] for node in module["node_ids"]]
    assert len(all_nodes) == len(set(all_nodes)) == 108
    assert set(all_nodes) == set(membership)
