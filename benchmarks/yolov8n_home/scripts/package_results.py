"""打包已完成的实验，并逐项回读 ZIP 校验 SHA-256。"""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root.parent / 'RK3588_YOLOv8n_首轮诊断复现包_20261106.zip'
if archive.exists():
    raise FileExistsError(f'交付包已存在，拒绝覆盖: {archive}')
folders = {'models', 'dataset', 'runs', 'scripts', 'tests', 'graphs', 'evidence', 'vendor'}
root_files = {'README.md', '实现计划.md', '首轮实测简报.md', 'config.json', 'summary.json', 'RK3588Bench.jsonl'}
files = sorted(p for p in root.rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
               and (p.relative_to(root).parts[0] in folders or p.relative_to(root).as_posix() in root_files)
               and not p.name.endswith(('.tar.gz', '.zip', '.pyc')))
manifest = {'run_id': 'yolov8n_home_20261106_01', 'quality_status': 'failed',
            'performance_scope': 'diagnostic_only', 'formal_acceptance': False,
            'file_count_excluding_manifest': len(files),
            'files': [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
    for path in files:
        z.write(path, arcname='yolov8n_home/' + path.relative_to(root).as_posix())
    z.writestr('yolov8n_home/package_manifest.json', manifest_bytes)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist()) == len(files) + 1
    assert len(set(z.namelist())) == len(z.namelist())
    reread = json.loads(z.read('yolov8n_home/package_manifest.json'))
    for item in reread['files']:
        data = z.read('yolov8n_home/' + item['path'])
        assert len(data) == item['bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    audit = json.loads(z.read('yolov8n_home/evidence/artifact_audit.json'))
    for item in audit['artifacts']:
        assert hashlib.sha256(z.read('yolov8n_home/' + item['path'])).hexdigest() == item['sha256']
    records = [json.loads(line) for line in z.read('yolov8n_home/RK3588Bench.jsonl').decode('utf-8').splitlines()]
    assert len(records) == 40 and len({r['sample_id'] for r in records}) == 40
    assert sum(len(r['hotspots']) for r in records) == 160
result = {'archive': str(archive.resolve()), 'bytes': archive.stat().st_size,
          'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'file_count_including_manifest': len(files) + 1, 'verified': True,
          'sample_count': 40, 'hotspot_count': 160, 'quality_status': 'failed',
          'scope': '逐项回读ZIP文件，校验文件长度、SHA-256、证据索引和样本/热点数量'}
(root.parent / 'RK3588_YOLOv8n_交付包核验_20261106.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
