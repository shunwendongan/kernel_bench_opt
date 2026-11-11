#!/usr/bin/env python3
"""Generate the machine-readable records of this branch (candidate.json, experiments/index.json,
workload.json, benchmarks/history) from one table, hashing the delivered source files.
Re-run after changing candidate sources; the records follow main/schemas/*.schema.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'yolov8n-COCO'
DATE = '20261111-120000'
SOURCE_BASE_COMMIT = 'c8ac23a29d372ecce00701e6afe7f980abad1bfe'

KERNEL_COMMON = ['kernels/common/fp16.h', 'kernels/common/tensor_view.h']
POST = ['kernels/post-dfl-nms-neon/yolo_post.h', 'kernels/post-dfl-nms-neon/yolo_post.c', 'kernels/post-dfl-nms-neon/bench_post.c']
PRE = ['kernels/pre-letterbox-neon/letterbox.h', 'kernels/pre-letterbox-neon/letterbox.c', 'kernels/pre-letterbox-neon/bench_letterbox.c']
BRIDGE = ['src/apps/rknn_bridge_v2.cpp', 'src/apps/rknn_bridge_v2.py', 'src/apps/build_bridge_v2.sh']
GRAPH = ['tools/npu/rewrite_yolov8n_graph.py', 'tools/npu/convert_variant.py', 'tools/verify/compare_onnx_variants.py']
VERIFY = ['tools/verify/l2_compare_bridges.py', 'tools/verify/l3_e2e_v2.py', 'scripts/run_yolo_board.sh']
WRAP = ['tools/yolo_kernels.py', 'scripts/build_host_kernels.sh']

CANDIDATES = [
    dict(id='head-logits6', kind='subgraph', parent='yolov8n-fp16-cpu-sigmoid-1b213f80', status='source_delivered', principle='01',
         hypothesis='Removing the three Sigmoid/ReduceSum/Clip tails and exporting class logits deletes the 24.5 ms CPU Sigmoid inside rknn_run; the host applies float32 sigmoid only to the per-anchor maximum logit, and canonical9 reconstructed with float32 sigmoid still passes the frozen numeric gates.',
         sources=GRAPH + POST + KERNEL_COMMON, runs=[f'{DATE}-head-logits6-l2', f'{DATE}-yolov8n-v2-combined-l3'],
         limitations=['RKNN conversion not yet executed (needs rknn-toolkit2 2.3.2 on Linux x86)', 'native output layout of the 6-output model unknown until queried on the board', 'no RK3588 timing yet'],
         evidence=['experiments/head-logits6/evidence/compare_logits6.json', 'experiments/head-logits6/evidence/compare_logits6_splitfree.json']),
    dict(id='io-zerocopy-fp16', kind='adapter', parent='rknn-bridge-v1-inputs-set-outputs-get', status='source_delivered', principle='02',
         hypothesis='Binding input/output DMA memory once with rknn_set_io_mem, writing the fp16 input in the native format and reading fp16 outputs in place removes the 13.2 ms inputs_set, 12.0 ms outputs_get and 1.7 ms copy of the first-round bridge without changing rknn_run or the exported float32 outputs.',
         sources=BRIDGE + PRE + POST + KERNEL_COMMON, runs=[f'{DATE}-io-zerocopy-fp16-l2', f'{DATE}-yolov8n-v2-combined-l3'],
         limitations=['compiled only with -fsyntax-only against the vendored rknn_api.h; not yet linked on the board', 'pass-through native input format and folded normalization are inferred from runtime metadata and must be confirmed bit-exactly in L2'],
         evidence=[]),
    dict(id='post-dfl-nms-neon', kind='kernel', parent='yolo-common-postprocess-numpy', status='source_delivered', principle='03',
         hypothesis='One NEON pass over the fp16 output buffers (max/argmax with exact score-sum prefilter, DFL with numpy reduction order, class-wise NMS with numpy ordering rules) reproduces yolo_common.postprocess within 2e-3 px and replaces 9.9 ms of numpy with about 1 ms on Cortex-A76.',
         sources=POST + KERNEL_COMMON + WRAP, runs=[f'{DATE}-host-kernels-l1', f'{DATE}-yolov8n-v2-combined-l3'],
         limitations=['Apple M4 timings only indicate direction', 'NC1HWC2 fast path assumes C2 == 8 (fp16); other block sizes fall back to the portable path'],
         evidence=[]),
    dict(id='pre-letterbox-neon', kind='kernel', parent='yolo-common-letterbox-opencv', status='source_delivered', principle='04',
         hypothesis='A fixed-point letterbox that reproduces OpenCV INTER_LINEAR byte for byte, fused with 114 padding, BGR->RGB and fp16 NHWC packing into the input DMA buffer, keeps the frozen input hashes while removing the OpenCV passes and the runtime uint8->fp16 conversion.',
         sources=PRE + KERNEL_COMMON + WRAP, runs=[f'{DATE}-host-kernels-l1', f'{DATE}-yolov8n-v2-combined-l3'],
         limitations=['byte equality verified against opencv-python 4.11 arm64 on the dev machine; board OpenCV build to be confirmed via frozen input hashes', 'horizontal pass is scalar int32 code'],
         evidence=[]),
    dict(id='c2f-split-free', kind='subgraph', parent='yolov8n-fp16-cpu-sigmoid-1b213f80', status='source_delivered', principle='05',
         hypothesis='Replacing every C2f cv1->Split with two half-width 1x1 convolutions (sliced weights) removes the eight Split copy nodes (2.86 ms node time) with bit-identical outputs; the extra read of the cv1 input costs less than the Split read+write.',
         sources=GRAPH, runs=[f'{DATE}-c2f-split-free-l2'],
         limitations=['RKNN conversion not yet executed', 'net gain depends on the compiler scheduling of the two half convolutions'],
         evidence=['experiments/c2f-split-free/evidence/compare_splitfree.json']),
    dict(id='stem-space-to-depth', kind='subgraph', parent='yolov8n-fp16-cpu-sigmoid-1b213f80', status='source_delivered', principle='06',
         hypothesis='Rewriting the 3-channel 3x3/s2 stem as a 12-channel 2x2/s1 convolution on the space-to-depth tensor (graph node or host packing) lowers the 4.1 ms, 0.7 % MAC-usage model.0 node by improving channel alignment and access pattern.',
         sources=GRAPH, runs=[f'{DATE}-stem-space-to-depth-l2'],
         limitations=['RKNN support/efficiency of SpaceToDepth and 2x2 kernels unknown', 'host variant changes the input contract; NEON s2d packing not implemented yet', 'float32 max abs 4.0e-5 from reordered accumulation'],
         evidence=['experiments/stem-space-to-depth/evidence/compare_s2dgraph.json', 'experiments/stem-space-to-depth/evidence/compare_s2dhost.json']),
    dict(id='npu-core-mask-sweep', kind='system', parent='rknn-bridge-v1-core-mask-7', status='proposed', principle='07',
         hypothesis='With core_mask=7 the PERF_DETAIL WorkLoad column shows every ConvExSwish on one core; rknn_run with core_mask=1 differs from 7 only by the Add/Concat/Conv/ConvClip nodes, so three cores do not shorten single-frame latency for this model.',
         sources=['src/apps/rknn_bridge_v2.cpp', 'tools/verify/l2_compare_bridges.py'], runs=[f'{DATE}-npu-core-mask-sweep-l2'],
         limitations=['observation only; no board measurement'], evidence=[]),
]

RUNS = {
    f'{DATE}-host-kernels-l1': dict(task='host-kernels', level='l1', candidates=['post-dfl-nms-neon', 'pre-letterbox-neon'],
                                     command='scripts/run_yolo_board.sh l1', inputs='synthetic tensors/images + frozen ONNX outputs',
                                     metrics=['bench_post ms (NEON vs reference)', 'bench_letterbox ms', 'pytest results']),
    f'{DATE}-io-zerocopy-fp16-l2': dict(task='io-zerocopy-fp16', level='l2', candidates=['io-zerocopy-fp16', 'post-dfl-nms-neon', 'pre-letterbox-neon'],
                                         command='scripts/run_yolo_board.sh l2 ${DEV_ROOT}/models/yolov8n_fp16.rknn ${OUT}', inputs='40 frozen inputs (dataset/inputs/*.u8), first-round .rknn',
                                         metrics=['pack_ms', 'run_ms', 'post_ms', 'wall_ms', 'outputs_bit_identical_to_v1', 'letterbox_hash_matches']),
    f'{DATE}-head-logits6-l2': dict(task='head-logits6', level='l2', candidates=['head-logits6'],
                                     command='scripts/run_yolo_board.sh l2 models/variants/yolov8n_logits6.rknn ${OUT}', inputs='40 frozen inputs, logits6 .rknn, frozen ONNX for gates',
                                     metrics=['run_ms vs final model', 'compare_canonical gates', 'detection matching', 'PERF_DETAIL Sigmoid rows absent']),
    f'{DATE}-c2f-split-free-l2': dict(task='c2f-split-free', level='l2', candidates=['c2f-split-free'],
                                       command='scripts/run_yolo_board.sh l2 models/variants/yolov8n_splitfree.rknn ${OUT}', inputs='40 frozen inputs, split-free .rknn',
                                       metrics=['run_ms', 'per-module node time (PERF_DETAIL)', 'compare_canonical gates']),
    f'{DATE}-stem-space-to-depth-l2': dict(task='stem-space-to-depth', level='l2', candidates=['stem-space-to-depth'],
                                            command='scripts/run_yolo_board.sh l2 models/variants/yolov8n_s2dgraph.rknn ${OUT}', inputs='40 frozen inputs, s2d .rknn',
                                            metrics=['model.0 node time', 'SpaceToDepth node time', 'compare_canonical gates']),
    f'{DATE}-npu-core-mask-sweep-l2': dict(task='npu-core-mask-sweep', level='l2', candidates=['npu-core-mask-sweep'],
                                            command='YOLO_CONFIGS="--config core7:1,2,0,7 --config core0:1,2,0,1 --config auto:1,2,0,0" scripts/run_yolo_board.sh l2 ${MODEL} ${OUT}',
                                            inputs='40 frozen inputs, same .rknn', metrics=['run_ms per core_mask', 'WorkLoad columns (detail mode)', 'outputs bit-identical across masks']),
    f'{DATE}-yolov8n-v2-combined-l3': dict(task='yolov8n-v2-combined', level='l3', candidates=['head-logits6', 'io-zerocopy-fp16', 'post-dfl-nms-neon', 'pre-letterbox-neon'],
                                            command='scripts/run_yolo_board.sh l3 ${MODEL} ${OUT}', inputs='40 frozen images x 10 repeats, seed 42',
                                            metrics=['decode_ms', 'letterbox_ms', 'pack_ms', 'run_ms', 'post_ms', 'e2e_ms (mean/median/p95)']),
}


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def dump(path, data):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    index = {'schema_version': 1, 'record_type': 'candidate_index', 'branch': BRANCH,
             'notes': 'One entry per experiments/<candidate-id>/candidate.json. Status values follow main/docs/CONVENTIONS.md section 4. Regenerate with tools/make_records.py.',
             'candidates': []}
    for c in CANDIDATES:
        cid = c['id']
        design = f'experiments/{cid}/design.md'
        principle = f"principles/{c['principle']}-{cid}.md"
        for p in [design, principle] + c['sources'] + c['evidence']:
            if not (ROOT / p).exists():
                raise FileNotFoundError(p)
        record = {'schema_version': 2, 'record_type': 'candidate', 'candidate_id': cid, 'candidate_kind': c['kind'],
                  'parent_candidate_id': c['parent'], 'status': c['status'],
                  'status_detail': 'delivered 2026-11-11 from the hotspot analysis of the first-round and final-model evidence; no RK3588 run yet',
                  'hypothesis': c['hypothesis'], 'source_base_commit': SOURCE_BASE_COMMIT,
                  'target': {'model': 'YOLOv8n Rockchip export, FP16, 640x640, batch 1', 'device': 'RK3588 (3x NPU, Cortex-A76/A55)',
                             'runtime': 'RKNN 2.3.2 / driver 0.9.8', 'baseline_evidence': ['benchmarks/yolov8n_home/summary.json',
                             'benchmarks/yolov8n_general/evidence/model/final_cpu_placement.perf.txt']},
                  'source_files': sorted(({'path': p, 'sha256': sha256(p)} for p in set(c['sources'])), key=lambda r: r['path']),
                  'dev_evidence': [{'path': p, 'sha256': sha256(p)} for p in c['evidence']],
                  'design': design, 'principle': principle, 'runs': c['runs'], 'limitations': c['limitations']}
        dump(f'experiments/{cid}/candidate.json', record)
        sources_dirs = sorted({p.rsplit('/', 1)[0] + '/' for p in c['sources']})
        index['candidates'].append({'candidate_id': cid, 'candidate_kind': c['kind'], 'parent_candidate_id': c['parent'],
                                    'status': c['status'], 'candidate_json': f'experiments/{cid}/candidate.json', 'design': design,
                                    'principle': principle, 'sources': sources_dirs, 'overlay_sets': [], 'runs': c['runs']})
    dump('experiments/index.json', index)
    dump('workload.json', {'schema_version': 1, 'record_type': 'workload_branch', 'branch': BRANCH, 'branch_kind': 'workload',
                           'model': {'name': 'YOLOv8n', 'deployment': 'Rockchip rknn_model_zoo detection export (9 outputs)',
                                     'onnx_sha256': '0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5',
                                     'rknn_sha256': {'first_round_npu_sigmoid': '56808e7b2d85de3cc59e953711d158f15b36bb4b33f12a05d8bd27e0a70d5813',
                                                     'final_cpu_sigmoid': '1b213f80a92b42a9d0709e4fabf77a7e04521cd52a6461be704d627bb9bba3e5'}},
                           'dataset': {'name': 'COCO', 'split': 'val2017', 'development_images': 40, 'acceptance_images': 4960, 'performance_images': 80},
                           'engines': [{'name': 'RKNN', 'runtime': '2.3.2', 'driver': '0.9.8'}],
                           'devices': ['RK3588 / 3x NPU / Cortex-A76+A55'],
                           'candidate_index': 'experiments/index.json', 'run_index': 'benchmarks/history/index.json',
                           'conventions': 'main branch docs/CONVENTIONS.md (legacy benchmark layout under benchmarks/ kept as is)',
                           'shared_tooling_branch': 'main',
                           'legacy_layout': ['benchmarks/yolov8n_home', 'benchmarks/yolov8n_general', 'benchmarks/presentation', 'benchmarks/fixtures', '汇报材料']})
    run_index = {'schema_version': 2, 'record_type': 'run_index', 'branch': BRANCH,
                 'notes': 'Planned board runs for the 2026-11-11 candidates; completed runs add summary.json/validation.json/manifest.json next to run.json.',
                 'entries': []}
    for run_id, r in RUNS.items():
        primary = r['candidates'][0]
        record = {'schema_version': 1, 'record_type': 'planned_run', 'task_id': None, 'run_id': run_id,
                  'candidate_id': primary, 'candidates': r['candidates'],
                  'parent_candidate_id': next(c['parent'] for c in CANDIDATES if c['id'] == primary), 'hypothesis_id': None,
                  'code': {'git_commit': None, 'source_base_commit': SOURCE_BASE_COMMIT, 'patch_sha256': None, 'source_dirty': None,
                           'command': r['command']},
                  'environment': {'device_id': None, 'soc': 'RK3588', 'os': None, 'kernel': None, 'runtime': 'RKNN 2.3.2 expected',
                                  'driver': '0.9.8 expected', 'container_image_id': None, 'container_packages_sha256': None,
                                  'frequency_temperature_load_file': None, 'frequency_locked': False},
                  'assets': {'model_sha256_manifest': None, 'dataset_sha256_manifest': 'benchmarks/yolov8n_home/dataset/samples.json (frozen 40 inputs)',
                             'sample_ids': [], 'capture_sha256_manifest': None, 'split_group': 'yolov8n_home_development40',
                             'inputs': r['inputs']},
                  'measurement': {'level': r['level'], 'phase': None, 'timing_domain': 'host perf_counter_ns + C++ steady_clock API spans + RKNN PERF_RUN/PERF_DETAIL',
                                  'unit': 'ms (perf_run in us)', 'profiling_enabled': None, 'warmup': 3,
                                  'repeats': 10 if r['level'] == 'l3' else 3, 'seed': 42, 'command_file': None, 'raw_file': None,
                                  'correctness_file': None, 'summary_file': None, 'metrics': r['metrics']},
                  'result': {'status': 'planned', 'correctness_passed': None, 'old_median': None, 'candidate_median': None,
                             'ratio_of_medians': None, 'failure_reason': None, 'notes': 'registered 2026-11-11; not executed'},
                  'analysis': {'profile_run_id': None, 'mapping_file': None, 'mapping_status': None, 'source_location': None,
                               'design_file': f'experiments/{primary}/design.md', 'decision': None},
                  'data_reuse': {'source_kind': 'planned', 'review_status': 'unreviewed', 'training_split': None,
                                 'license_provenance': 'COCO val2017 CC BY 4.0 annotations; images per COCO licence metadata',
                                 'known_limitations': ['frequency not locked', 'development images only']}}
        dump(f'benchmarks/history/{run_id}/run.json', record)
        run_index['entries'].append({'run_id': run_id, 'candidate_id': primary, 'task_id': None, 'level': r['level'],
                                     'status': 'planned', 'status_detail': 'registered 2026-11-11; not executed',
                                     'path': f'benchmarks/history/{run_id}', 'run_record': f'benchmarks/history/{run_id}/run.json',
                                     'report': None, 'assets': None, 'legacy_id': False})
    dump('benchmarks/history/index.json', run_index)
    print(f'wrote {len(CANDIDATES)} candidates, {len(RUNS)} planned runs')


if __name__ == '__main__':
    main()
