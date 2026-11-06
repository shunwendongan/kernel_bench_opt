"""把已校验统计与真实示例输入嵌入可离线浏览的汇报。"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "汇报材料" / "2026-11-06_YOLOv8n_RK3588_小组汇报.html"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def image_uri(raw: bytes, kind: str) -> str:
    return f"data:image/{kind};base64," + base64.b64encode(raw).decode("ascii")


def build(output: Path) -> dict:
    source = HERE / "report_data.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["validation"]["status"] != "passed":
        raise ValueError("源数据未通过校验，拒绝生成展示")
    if data["quality"]["formal_acceptance"]:
        raise ValueError("本模板只适用于尚未正式验收的诊断与开发结果")
    # UI 别名只整理字段位置，不改变原统计数值。
    payload = copy.deepcopy(data)
    payload["baseline"].update(data["baseline"]["stages_ms"])
    payload["baseline"]["api"] = data["baseline"]["api_ms"]
    payload["quality"]["home"]["reference_match_fraction"] = data["quality"]["home"]["detection_matching"]["reference_fraction"]
    payload["hotspots"]["frequency"] = data["hotspots"]["selection_stats"]
    payload["hotspots"]["representative"]["items"] = data["hotspots"]["representative"]["hotspots"]
    general = data["datasets"]["general"]
    payload["datasets"]["general"]["downloaded_count"] = general["download"]["logged_success_count"]
    payload["datasets"]["general"]["profile_category_count"] = general["profile_subset"]["actual_category_count"]
    payload["meta"]["model_sha256"] = data["meta"]["model"]["rknn_sha256"]
    payload["meta"]["validated_reports"] = data["validation"]["strictly_mapped_reports"]
    selection_sources = HERE / "selection_sources.json"
    payload["meta"]["selection_sources"] = json.loads(selection_sources.read_text(encoding="utf-8"))
    payload["meta"]["sources"] = [{"path": path, **record} for path, record in data["meta"]["sources"].items()]
    manifest = json.loads((ROOT / "benchmarks/yolov8n_home/dataset/samples.json").read_text(encoding="utf-8"))
    example = next(s for s in manifest["samples"] if s["id"] == 312192)
    home = ROOT / "benchmarks/yolov8n_home"
    original = (home / example["image"]).read_bytes()
    rgb = (home / example["input"]).read_bytes()
    if sha256(original) != example["image_sha256"] or sha256(rgb) != example["input_sha256"]:
        raise ValueError("原图或真实 RGB 输入哈希变化")
    w, h, channels = example["input_shape"]
    if channels != 3 or len(rgb) != w * h * channels:
        raise ValueError("RGB 输入字节数不符合冻结形状")
    image = Image.frombytes("RGB", (w, h), rgb)
    png = io.BytesIO()
    image.save(png, format="PNG", optimize=True)
    payload["meta"]["image_credit"] = (
        "示例图：COCO val2017，image_id=312192，license=4 / CC BY 2.0。"
        "原始 Flickr 记录：" + example["flickr_url"] + "。作者名未保存在原始 COCO 清单中。"
        "右图为冻结 RGB 输入的无损 PNG，变换仅为上下各补 80 像素。"
    )
    payload["meta"]["presentation"] = {
        "data_sha256": sha256(source.read_bytes()),
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "template_sha256": sha256((HERE / "yolo_report.template.html").read_bytes()),
        "selection_sources_sha256": sha256(selection_sources.read_bytes()),
        "image_id": example["id"],
        "original_sha256": sha256(original),
        "input_sha256": sha256(rgb),
        "input_png_sha256": sha256(png.getvalue()),
    }
    html = (HERE / "yolo_report.template.html").read_text(encoding="utf-8")
    replacements = {
        "__REPORT_JSON__": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
        "__ORIGINAL_IMAGE__": image_uri(original, "jpeg"),
        "__INPUT_IMAGE__": image_uri(png.getvalue(), "png"),
    }
    for marker, value in replacements.items():
        if html.count(marker) != 1:
            raise ValueError(f"模板占位符数量错误：{marker}")
        html = html.replace(marker, value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return {"output": str(output), "bytes": output.stat().st_size,
            "sha256": sha256(output.read_bytes()), "data_validation": data["validation"]["status"],
            "nodes": len(data["detail"]["nodes"]), "edges": len(data["detail"]["edges"]),
            "modules": len(data["detail"]["modules"]), "reports": data["detail"]["report_count"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
