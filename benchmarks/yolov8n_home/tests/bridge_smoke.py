"""仅用全 114 RGB 合成输入验证真实 RKNN 桥接；不作为真图性能或精度结果。

板端：python tests/bridge_smoke.py --root /path/to/yolov8n_home
输出 evidence/bridge_smoke_*，默认拒绝覆盖已有证据，可用 --prefix 指定新前缀。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rknn_bridge import RknnSession

ONNX_SOURCE_SHA256 = "0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_verified(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"证据回读不一致：{path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--prefix", default="bridge_smoke")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix or not args.prefix.startswith("bridge_smoke"):
        raise ValueError("prefix must be a filename beginning with bridge_smoke")
    root = args.root.resolve()
    model = root / "models/yolov8n_fp16.rknn"
    library = (args.library or root / "scripts/librknn_bridge.so").resolve()
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    paths = {name: evidence / f"{args.prefix}_{suffix}" for name, suffix in {
        "summary": "summary.json", "baseline": "baseline.npz", "detail": "detail.npz",
        "perf": "detail.perf.txt", "invalid": "invalid_path.json",
    }.items()}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"保留既有证据，请改 --prefix：{path}")

    missing = model.with_name(".__bridge_smoke_missing__.rknn")
    if missing.exists():
        raise FileExistsError(missing)
    try:
        unexpected = RknnSession(missing, library)
    except RuntimeError as error:
        invalid = {"passed": "rknn_init" in str(error), "error": str(error),
                   "model_path": str(missing), "library_sha256": sha256(library)}
        write_json_verified(paths["invalid"], invalid)
        if not invalid["passed"]:
            raise
    else:
        unexpected.close()
        raise AssertionError("不存在的模型必须初始化失败")

    rgb = np.full((640, 640, 3), 114, dtype=np.uint8)
    with RknnSession(model, library, detail=False, core_mask=7) as session:
        baseline_metadata = session.metadata
        baseline, baseline_metrics = session.run(rgb)
        repeat, repeat_metrics = session.run(rgb)
        for first, second in zip(baseline, repeat):
            np.testing.assert_array_equal(first, second)
    with RknnSession(model, library, detail=True, core_mask=7) as session:
        detail_metadata = session.metadata
        detail, detail_metrics = session.run(rgb, perf_path=paths["perf"])

    if len(baseline) != 9 or len(detail) != 9:
        raise AssertionError("必须返回全部 9 个输出")
    rtol, atol = 1e-5, 1e-6
    comparisons = []
    for index, (base, detailed) in enumerate(zip(baseline, detail)):
        expected = tuple(baseline_metadata["outputs"][index]["dims"])
        if base.shape != expected or detailed.shape != expected:
            raise AssertionError(f"输出 {index} 形状不匹配")
        if base.dtype != np.float32 or detailed.dtype != np.float32:
            raise AssertionError("want_float=1 必须返回 float32")
        if not np.isfinite(base).all() or not np.isfinite(detailed).all():
            raise AssertionError(f"输出 {index} 含非有限值")
        np.testing.assert_allclose(base, detailed, rtol=rtol, atol=atol)
        comparisons.append({"index": index, "shape": list(base.shape), "finite": True,
                            "max_abs_difference": float(np.max(np.abs(base - detailed))),
                            "exact_equal": bool(np.array_equal(base, detailed)),
                            "baseline_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
                            "detail_sha256": hashlib.sha256(detailed.tobytes()).hexdigest()})
    report = paths["perf"].read_bytes()
    if not report.strip(b"\x00 \r\n\t"):
        raise AssertionError("详细报告为空")
    for mode, arrays in (("baseline", baseline), ("detail", detail)):
        np.savez_compressed(paths[mode], **{f"output_{i}": a for i, a in enumerate(arrays)})
        with np.load(paths[mode]) as saved:
            if len(saved.files) != len(arrays):
                raise AssertionError("输出证据数量回读不一致")
            for i, array in enumerate(arrays):
                np.testing.assert_array_equal(saved[f"output_{i}"], array)

    summary = {
        "passed": True,
        "purpose": "仅合成输入接口烟测；不是40张真图基准或检测精度结果",
        "input": {"kind": "synthetic_constant", "value": 114, "dtype": "uint8",
                  "shape": list(rgb.shape), "color": "RGB",
                  "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()},
        "onnx_source_sha256": ONNX_SOURCE_SHA256,
        "rknn_sha256": sha256(model), "library_sha256": sha256(library),
        "rtol": rtol, "atol": atol,
        "baseline": {"metadata": baseline_metadata, "timings": baseline_metrics,
                     "resident_repeat_timings": repeat_metrics, "repeat_exact_equal": True},
        "detail": {"metadata": detail_metadata, "timings": detail_metrics},
        "outputs": comparisons,
        "perf_report_bytes": len(report), "perf_report_sha256": sha256(paths["perf"]),
        "artifacts": {name: {"path": str(path.relative_to(root)), "sha256": sha256(path)}
                      for name, path in paths.items() if name != "summary"},
    }
    write_json_verified(paths["summary"], summary)
    print(json.dumps({"passed": True, "purpose": summary["purpose"],
                      "outputs": comparisons, "perf_report_bytes": len(report),
                      "sdk_version": baseline_metadata["sdk_version"],
                      "mem_size": baseline_metadata["mem_size"],
                      "baseline_timings": baseline_metrics, "detail_timings": detail_metrics,
                      "summary": str(paths["summary"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
