"""从实测记录核验覆盖、汇总性能并生成 RK3588Bench 索引。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_measurements(rows, image_ids, repeats):
    expected = {(ident, repeat) for ident in image_ids for repeat in range(repeats)}
    actual = [(row['image_id'], row['repeat']) for row in rows]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('执行记录缺失、重复或超出冻结样本范围')
    if any(r['status'] not in ('success', 'failed') for r in rows):
        raise ValueError('执行状态无效')
    if any(r['status'] == 'failed' and not r.get('error') for r in rows):
        raise ValueError('失败记录缺少原因')
    counts = Counter(r['status'] for r in rows)
    return {'expected': len(expected), 'success': counts['success'], 'failed': counts['failed']}


def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('统计值必须是非负有限数')
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p95': float(np.percentile(a, 95)),
            'min': float(a.min()), 'max': float(a.max()), 'std_population': float(a.std())}


def check_artifact(root, relative, expected=None):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('证据路径越出实验目录')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f'证据哈希不一致: {relative}')
    return {'path': path.relative_to(root).as_posix(), 'sha256': actual, 'bytes': path.stat().st_size}


def assemble_sample(sample, config, graph, executions, hotspots, quality_status,
                    quality_evidence='runs/verify_01/verification.json'):
    status = 'success' if all(e['status'] == 'success' for e in executions) else 'failed'
    return {
        'sample_id': sample['sample_id'], 'model': config['model'],
        'model_sha256': config['rknn_sha256'], 'model_graph': graph['path'],
        'model_graph_sha256': graph['sha256'],
        'source': {'dataset': 'COCO val2017', 'image_id': sample['id'],
                   'source_record': sample, 'reference_annotations': 'dataset/instances_home40.json',
                   'annotation_ids': sample['annotation_ids'], 'scene_review': 'dataset/scene_review.json'},
        'input': {'image': sample['image'], 'image_sha256': sample['image_sha256'],
                  'tensor': sample['input'], 'sha256': sample['input_sha256'],
                  'shape': sample['input_shape'], 'dtype': 'uint8', 'layout': 'HWC',
                  'colors': 'RGB', 'transform': sample['transform']},
        'run': {'id': config['run_id'], 'artifact_root': '.', 'config': 'config.json',
                'performance_scope': 'diagnostic_only' if quality_status != 'passed' else 'quality_checked'},
        'status': status, 'status_semantics': '仅表示实测执行状态，质量另列',
        'quality': {'status': quality_status, 'evidence': quality_evidence},
        'executions': executions,
        'hotspots': [dict(h, id=f"{config['run_id']}:{sample['sample_id']}:h{h['rank']}") for h in hotspots],
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize_resources(snapshots):
    observations = defaultdict(list)
    for snapshot in snapshots:
        for path, text in snapshot['sysfs'].items():
            if text is not None and (path.endswith('/temp') or path.endswith('/cur_freq') or path.endswith('/scaling_cur_freq')):
                observations[path].append(float(text))
    return {'observed_values': {key: sorted(set(values)) for key, values in observations.items()},
            'snapshot_count': len(snapshots), 'frequency_locked': False,
            'process_max_rss_kib': max(s['process_max_rss_kib'] for s in snapshots),
            'power_watts': None, 'energy_joules': None,
            'scope': '离散快照，不能证明整段运行的频率或温度恒定'}


def analyze(root, baseline, detail, verify='runs/verify_01'):
    from graph_tools import extract_graph, parse_perf, bind_graph, select_hotspots
    root = Path(root).resolve()
    config, manifest = read_json(root / 'config.json'), read_json(root / 'dataset/samples.json')
    quality_path = f'{verify}/verification.json'
    quality = read_json(root / quality_path)
    policy = read_json(root / 'evidence/diagnostic_policy.json')
    if quality['status'] != 'passed' and policy['performance_scope'] != 'diagnostic_only':
        raise ValueError('质量未通过，只能生成诊断结果')
    artifacts = {}

    def evidence(path, expected=None):
        item = check_artifact(root, path, expected)
        artifacts[item['path']] = item
        return item

    config_hash = evidence('config.json')['sha256']
    samples_hash = evidence('dataset/samples.json')['sha256']
    evidence('models/yolov8n_fp16.rknn', config['rknn_sha256'])
    evidence('models/yolov8n.onnx', config['onnx_sha256'])
    evidence(manifest['subset_annotation'], manifest['subset_annotation_sha256'])
    evidence(quality_path)
    for name, expected in quality['evidence'].items():
        evidence(f'{verify}/{name}', expected)
    evidence('dataset/scene_review.json')
    evidence('evidence/diagnostic_policy.json')
    ids = [s['id'] for s in manifest['samples']]
    runs, counts, resources, runtimes = {}, {}, {}, {}
    by_sample = defaultdict(list)
    for mode, run_dir, repeats in [('verify', verify, 1),
                                  ('baseline', baseline, config['baseline']['repeats_per_sample']),
                                  ('detail', detail, 1)]:
        path = root / run_dir
        rows = read_jsonl(path / 'measurements.jsonl')
        counts[mode] = validate_measurements(rows, ids, repeats)
        start, completion = read_json(path / 'start.json'), read_json(path / 'completion.json')
        runtime = read_json(path / 'runtime.json')
        if start['config_sha256'] != config_hash or start['samples_sha256'] != samples_hash:
            raise ValueError(f'{mode}: 运行配置或样本哈希不一致')
        if completion['expected_measurements'] != len(rows) or completion['failed'] != counts[mode]['failed']:
            raise ValueError(f'{mode}: completion 声明与记录不一致')
        if runtime['metadata']['detail'] != (mode == 'detail') or runtime['metadata']['core_mask'] != 7:
            raise ValueError(f'{mode}: 运行标志或核配置不符合协议')
        for name, sha in start['source_sha256'].items():
            evidence('scripts/' + name, sha)
        for filename in ('measurements.jsonl', 'start.json', 'completion.json', 'runtime.json', 'warmup.json', 'resources.jsonl'):
            evidence(f'{run_dir}/{filename}')
        for row in rows:
            if row['mode'] != mode or row['sample_id'] != next(s['sample_id'] for s in manifest['samples'] if s['id'] == row['image_id']):
                raise ValueError('执行记录的模式或样本标识不符')
            for key in ('output', 'detections', 'perf'):
                if row.get(key):
                    evidence(row[key], row[key + '_sha256'])
            if row['status'] == 'success':
                if row['repeat'] == 0 and (not row.get('output') or not row.get('detections')):
                    raise ValueError('首次正式执行缺少输出证据')
                times = [row[k] for k in ('start_ns', 'preprocessed_ns', 'inference_complete_ns', 'end_ns')]
                if times != sorted(times):
                    raise ValueError('主机时间戳不单调')
            if mode != 'verify':
                by_sample[row['image_id']].append(row)
        runs[mode] = rows
        runtimes[mode] = runtime
        snapshots = [start['resources']] + read_jsonl(path / 'resources.jsonl') + [completion['resources']]
        resources[mode] = summarize_resources(snapshots)
    native = extract_graph(root / 'models/yolov8n_fp16.rknn')
    detail_rows = {r['image_id']: r for r in runs['detail']}
    detail_parsed, hotspots_by_sample, graph_identity = {}, {}, None
    shared_graph = None
    for ident, execution in detail_rows.items():
        if execution['status'] != 'success':
            hotspots_by_sample[ident] = []
            continue
        if not execution.get('perf'):
            raise ValueError('详细执行成功但缺少原始节点报告')
        parsed = parse_perf(root / execution['perf'])
        bound = bind_graph(native, parsed)
        identity = [(r['id'], r['type'], r['name'], r['target']) for r in parsed]
        if graph_identity is None:
            graph_identity = identity
            shared_graph = bound
        elif identity != graph_identity:
            raise ValueError('样本间节点身份或设备映射发生变化')
        detail_parsed[ident] = parsed
        hotspots_by_sample[ident] = select_hotspots(bound, parsed)
        node_path = f'{detail}/nodes/{ident:012d}.json'
        save_json(root / node_path, parsed)
        execution['nodes'] = evidence(node_path)
    if shared_graph is None:
        raise ValueError('未取得任何可绑定的详细报告')
    # 共用节点表不附带任意样本的耗时，时间保留在各自 nodes 文件与 JSONL 热点中。
    for node in shared_graph['nodes']:
        for key in ('time_us', 'time_ms', 'performance'):
            node.pop(key, None)
    shared_graph['model_sha256'] = config['rknn_sha256']
    shared_graph['validated_reports'] = len(detail_parsed)
    save_json(root / 'graphs/compiled_graph.json', shared_graph)
    graph_ref = evidence('graphs/compiled_graph.json')
    records = []
    for sample in manifest['samples']:
        evidence(sample['image'], sample['image_sha256'])
        evidence(sample['input'], sample['input_sha256'])
        records.append(assemble_sample(sample, config, graph_ref, by_sample[sample['id']],
                                       hotspots_by_sample[sample['id']], quality['status'], quality_path))
    (root / 'RK3588Bench.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    evidence('RK3588Bench.jsonl')
    successful = [r for r in runs['baseline'] if r['status'] == 'success']
    normal_stats = {k: stats([r[k] for r in successful]) for k in ('preprocess_ms', 'inference_wall_ms', 'postprocess_ms', 'e2e_ms')}
    normal_stats['api'] = {k: stats([r['api'][k] for r in successful]) for k in successful[0]['api']} if successful else {}
    type_times, target_times, total_times = defaultdict(list), defaultdict(list), []
    type_counts, target_counts = Counter(), Counter()
    for parsed in detail_parsed.values():
        types, targets = defaultdict(float), defaultdict(float)
        for row in parsed:
            types[row['type']] += row['time_ms']
            targets[row['target']] += row['time_ms']
        for key, value in types.items():
            type_times[key].append(value)
        for key, value in targets.items():
            target_times[key].append(value)
        type_counts = Counter(r['type'] for r in parsed)
        target_counts = Counter(r['target'] for r in parsed)
        total_times.append(sum(r['time_ms'] for r in parsed))
    selected = defaultdict(list)
    for record in records:
        for hotspot in record['hotspots']:
            selected[hotspot['name']].append(hotspot)
    selected_summary = sorted([{'name': name, 'selected_sample_count': len(items),
                               'node_ids': items[0]['node_ids'], 'time_ms_when_selected': stats([h['time_ms'] for h in items])}
                              for name, items in selected.items()], key=lambda x: -x['time_ms_when_selected']['mean'])
    summary = {
        'run_id': config['run_id'], 'quality': quality, 'performance_scope': policy['performance_scope'],
        'formal_acceptance': False, 'counts': counts,
        'baseline': {'run': baseline, 'pooled_repetitions': normal_stats, 'resources': resources['baseline'],
                     'runtime': runtimes['baseline'], 'warmup': read_json(root / baseline / 'warmup.json'),
                     'per_sample': [{'image_id': ident, 'inference_wall_ms': stats([r['inference_wall_ms'] for r in successful if r['image_id'] == ident]),
                                     'e2e_ms': stats([r['e2e_ms'] for r in successful if r['image_id'] == ident])} for ident in ids]},
        'detail': {'run': detail, 'report_count': len(detail_parsed), 'node_count': len(shared_graph['nodes']),
                   'tensor_count': len(shared_graph['tensors']), 'edge_count': len(shared_graph['edges']),
                   'node_time_sum_ms': stats(total_times), 'target_counts': dict(target_counts),
                   'type_counts': dict(type_counts), 'type_time_ms': {k: stats(v) for k, v in type_times.items()},
                   'target_time_ms': {k: stats(v) for k, v in target_times.items()},
                   'hotspot_count': sum(len(r['hotspots']) for r in records), 'selected_hotspots': selected_summary,
                   'resources': resources['detail'], 'runtime': runtimes['detail'],
                   'time_semantics': '详细轮编译节点耗时之和，不是独立子图墙钟时间'},
        'unmeasured': {'internal_operator_api': None, 'kernel_timeline': None, 'power': None,
                       'subgraph_boundary_values': None, 'cross_backend_cost_matrix': None},
    }
    save_json(root / 'summary.json', summary)
    evidence('summary.json')
    save_json(root / 'evidence/artifact_audit.json', {'status': 'passed', 'sample_count': len(records),
              'counts': counts, 'quality_status': quality['status'], 'formal_acceptance': False,
              'hotspot_count': summary['detail']['hotspot_count'], 'artifact_count': len(artifacts),
              'artifacts': sorted(artifacts.values(), key=lambda x: x['path'])})
    print(json.dumps({'counts': counts, 'quality_status': quality['status'],
                      'baseline': normal_stats, 'detail_nodes': len(shared_graph['nodes']),
                      'hotspots': summary['detail']['hotspot_count'], 'artifacts': len(artifacts)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify', default='runs/verify_01')
    parser.add_argument('--baseline', default='runs/baseline_diagnostic_01')
    parser.add_argument('--detail', default='runs/detail_diagnostic_01')
    args = parser.parse_args()
    analyze(args.root, args.baseline, args.detail, args.verify)
