#!/usr/bin/env python3
"""Convert a rewritten YOLOv8n ONNX variant to RKNN with rknn-toolkit2 (Linux host).

Mirrors benchmarks/yolov8n_general/scripts/prepare_model.py::convert: FP16,
do_quantization=False, mean 0 / std 255, target rk3588. Differences:
  * the source is a variant produced by tools/npu/rewrite_yolov8n_graph.py whose
    sha256 must match the rewrite record (--record);
  * for nine-output variants the final Sigmoid outputs are still placed on the
    CPU with op_target (same as the frozen final model); logits6 variants have no
    Sigmoid left, so op_target is empty and nothing runs on the CPU inside rknn_run;
  * --host-input-shape declares the 12x320x320 input of a host space-to-depth
    variant (the runtime input attr then describes that tensor, not an image).
The conversion record (JSON) is written next to the output; existing outputs are
never overwritten so earlier evidence stays intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid_outputs(onnx_path):
    """Output tensor names produced by Sigmoid (op_target keys) in the variant."""
    import onnx
    graph = onnx.load(str(onnx_path)).graph
    producers = {o: n for n in graph.node for o in n.output}
    names = []
    for output in graph.output:
        node = producers.get(output.name)
        if node is not None and node.op_type == 'Sigmoid':
            names.append(output.name)
    return names


def convert(source, output, record_path, rewrite_record=None, cpu_sigmoid=True, verbose=True):
    source, output, record_path = Path(source), Path(output), Path(record_path)
    if output.exists() or record_path.exists():
        raise FileExistsError('use new output/record paths to preserve existing evidence')
    source_sha = digest(source)
    expected = None
    if rewrite_record:
        rewrite = json.loads(Path(rewrite_record).read_text(encoding='utf-8'))
        expected = rewrite.get('output_sha256')
        if expected and expected != source_sha:
            raise ValueError(f'variant sha256 {source_sha} does not match the rewrite record {expected}')
    keys = sigmoid_outputs(source) if cpu_sigmoid else []
    config = {'target_platform': 'rk3588', 'mean_values': [[0, 0, 0]], 'std_values': [[255, 255, 255]]}
    if keys:
        config['op_target'] = {k: 'cpu' for k in keys}
    record = {'schema_version': 1, 'record_type': 'rknn_conversion', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'toolkit2': None, 'source_sha256': source_sha, 'source_path': str(source), 'rewrite_record': rewrite_record,
              'output_path': str(output), 'script_sha256': digest(__file__), 'config': config,
              'do_quantization': False, 'calibration_dataset_used': False,
              'output_contract': 'logits6' if not keys and len(sigmoid_outputs(source)) == 0 else 'canonical9',
              'cpu_sigmoid_outputs': keys}
    from rknn.api import RKNN
    record['toolkit2'] = importlib.metadata.version('rknn-toolkit2')
    runtime = RKNN(verbose=verbose)
    try:
        for stage, action in (('config', lambda: runtime.config(**config)),
                              ('load_onnx', lambda: runtime.load_onnx(model=str(source))),
                              ('build', lambda: runtime.build(do_quantization=False)),
                              ('export_rknn', lambda: runtime.export_rknn(str(output)))):
            record['stage'] = stage
            code = action()
            record.setdefault('return_codes', {})[stage] = code
            if code != 0:
                raise RuntimeError(f'{stage} returned {code}')
        record.update(status='compiled', rknn_sha256=digest(output), bytes=output.stat().st_size)
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        runtime.release()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='variant ONNX from rewrite_yolov8n_graph.py')
    parser.add_argument('--output', required=True, help='new .rknn path (must not exist)')
    parser.add_argument('--record', required=True, help='new conversion record JSON path (must not exist)')
    parser.add_argument('--rewrite-record', help='rewrite record JSON; its output_sha256 must match --source')
    parser.add_argument('--no-cpu-sigmoid', action='store_true', help='leave remaining Sigmoid outputs on the NPU')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        convert(args.source, args.output, args.record, args.rewrite_record, not args.no_cpu_sigmoid, not args.quiet)
    except ImportError as exc:
        print(f'rknn-toolkit2 is not importable here: {exc}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
