"""分别验证合成数据的分层契约与真实 COCO 的冻结结果。"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'yolov8n_home'


@pytest.fixture(scope='session')
def coco():
    annotation_path = LEGACY / 'dataset/instances_val2017.json'
    # 仅缺失时跳过；文件存在时，读取和解析错误必须直接失败。
    if not annotation_path.exists():
        pytest.skip(f'缺少外部 COCO 标注：{annotation_path}')
    return json.loads(annotation_path.read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def dev_ids():
    old = json.loads((LEGACY / 'dataset/samples.json').read_text(encoding='utf-8'))
    return [row['id'] for row in old['samples']]


def module():
    path = ROOT / 'scripts/prepare_general.py'
    assert path.is_file(), 'prepare_general.py does not exist yet'
    spec = importlib.util.spec_from_file_location('prepare_general', path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def synthetic_coco():
    """仅用于单元测试，不作为真实 COCO 数据或实测证据。"""
    supercategories = ('person', 'vehicle', 'animal', 'sports', 'food', 'kitchen',
                       'furniture', 'electronic', 'outdoor', 'accessory', 'appliance', 'indoor')
    categories = [{'id': n, 'name': f'synthetic_{name}', 'supercategory': name}
                  for n, name in enumerate(supercategories, start=1)]
    # 各分层共用候选图片；开发集也带有效标注，必须主动排除。
    images = [{'id': n, 'file_name': f'synthetic_{n}.jpg',
               'width': 32, 'height': 24, 'license': 1 + n % 3}
              for n in range(1, 22)]
    annotations = [{'id': image['id'] * 100 + category['id'],
                    'image_id': image['id'], 'category_id': category['id'],
                    'iscrowd': int(image['id'] == 19),
                    'bbox': [0, 0, 0 if image['id'] == 20 else 10,
                             0 if image['id'] == 21 else 10]}
                   for image in images for category in categories]
    return {'images': images, 'annotations': annotations, 'categories': categories}


def test_synthetic_split_is_disjoint_and_keeps_stratum_quotas(synthetic_coco):
    split = module().make_split(synthetic_coco, [1, 2], per_stratum=2)
    dev, holdout = set(split['dev_ids']), set(split['acceptance_ids'])
    profile = split['profile_samples']
    selected = {row['id'] for row in profile}
    assert split['dev_ids'] == [1, 2]
    assert split['acceptance_ids'] == list(range(3, 22))
    assert not dev & holdout
    assert dev | holdout == {image['id'] for image in synthetic_coco['images']}
    assert len(profile) == len(selected) == 16
    assert selected <= holdout
    # 最后三张图片只含 crowd、零宽框或零高框，不能进入剖析集。
    assert selected == set(range(3, 19))
    expected_groups = {'person': [1], 'traffic_vehicles': [2], 'animals': [3],
                       'sports': [4], 'food_kitchen': [5, 6], 'furniture': [7],
                       'electronics': [8], 'other_objects': [9, 10, 11, 12]}
    assert Counter(row['stratum_name'] for row in profile) == {
        name: 2 for name in expected_groups}
    assert len(split['strata']) == len(expected_groups)
    assert {group['name']: group['category_ids'] for group in split['strata']} == expected_groups
    assert split['seed'] == 20261106 and split['per_stratum'] == 2


def test_synthetic_split_is_deterministic_and_order_independent(synthetic_coco):
    prep = module()
    split = prep.make_split(synthetic_coco, [1, 2], per_stratum=2)
    assert split == prep.make_split(synthetic_coco, [1, 2], per_stratum=2)
    reordered = {key: list(reversed(rows)) for key, rows in synthetic_coco.items()}
    assert split == prep.make_split(reordered, [2, 1], per_stratum=2)


def test_synthetic_frozen_split_cannot_replace_ids(tmp_path, synthetic_coco):
    prep = module()
    split = prep.make_split(synthetic_coco, [1, 2], per_stratum=2)
    path = tmp_path / 'dataset/split.json'
    prep.freeze_split(path, split)
    frozen_bytes = path.read_bytes()
    assert json.loads(frozen_bytes) == split
    prep.freeze_split(path, split)
    assert path.read_bytes() == frozen_bytes
    changed = prep.make_split(synthetic_coco, [2, 3], per_stratum=2)
    with pytest.raises(ValueError, match='frozen'):
        prep.freeze_split(path, changed)
    assert path.read_bytes() == frozen_bytes


@pytest.mark.parametrize('development_source', [
    'relocated/legacy/dataset/samples.json',
    'C:/relocated/legacy/dataset/samples.json',
])
def test_freeze_reuses_identity_across_locations(tmp_path, synthetic_coco, development_source):
    prep = module()
    frozen = {**prep.make_split(synthetic_coco, [1, 2], per_stratum=2),
              'annotation_sha256': 'a' * 64,
              'development_manifest_sha256': 'b' * 64,
              'development_source': 'original/legacy/dataset/samples.json'}
    path = tmp_path / 'relocated/root/dataset/split.json'
    path.parent.mkdir(parents=True)
    # Preserve an existing archive's formatting as well as its sampling identity.
    original = (json.dumps(frozen, ensure_ascii=False, indent=1) + '\r\n').encode('utf-8')
    path.write_bytes(original)
    original_hash = prep.sha256(path)
    proposed = {**frozen, 'development_source': development_source}

    actual = prep.freeze_split(path, proposed)

    assert actual == frozen
    assert actual != proposed
    assert path.read_bytes() == original
    assert prep.sha256(path) == original_hash


@pytest.mark.parametrize('field', [
    'annotation_sha256', 'development_manifest_sha256', 'seed', 'per_stratum',
    'strata', 'dev_ids', 'acceptance_ids', 'profile_samples', 'sampling_definition',
    'failure_policy', 'schema_version',
])
def test_freeze_rejects_identity_changes_after_relocation(tmp_path, synthetic_coco, field):
    prep = module()
    frozen = {**prep.make_split(synthetic_coco, [1, 2], per_stratum=2),
              'annotation_sha256': 'a' * 64,
              'development_manifest_sha256': 'b' * 64,
              'development_source': 'original/legacy/dataset/samples.json'}
    path = tmp_path / 'dataset/split.json'
    prep.freeze_split(path, frozen)
    original = path.read_bytes()
    original_hash = prep.sha256(path)
    proposed = json.loads(original)
    proposed['development_source'] = 'C:/relocated/legacy/dataset/samples.json'
    value = proposed[field]
    if isinstance(value, list):
        proposed[field] = value[:-1]
    elif isinstance(value, int):
        proposed[field] = value + 1
    else:
        proposed[field] = value + ' changed'

    with pytest.raises(ValueError, match='frozen'):
        prep.freeze_split(path, proposed)

    assert path.read_bytes() == original
    assert prep.sha256(path) == original_hash


@pytest.mark.parametrize('relative_legacy', [False, True])
def test_main_prepares_archived_split_after_relocation(tmp_path, synthetic_coco, monkeypatch,
                                                       relative_legacy):
    prep = module()
    original = tmp_path / 'original'
    root = original / 'general'
    legacy = original / 'legacy'
    source = {**synthetic_coco,
              'images': [{'id': n} for n in range(1, 5001)],
              'annotations': [{'image_id': n, 'category_id': category['id'],
                               'iscrowd': 0, 'bbox': [0, 0, 10, 10]}
                              for n in range(41, 121) for category in synthetic_coco['categories']]}
    prep.write_json(legacy / 'dataset/instances_val2017.json', source)
    old = {'samples': [{'id': n} for n in range(1, 41)]}
    prep.write_json(legacy / 'dataset/samples.json', old)
    monkeypatch.setattr(prep, 'ANNOTATION_SHA256', prep.sha256(legacy / 'dataset/instances_val2017.json'))
    monkeypatch.setattr(prep.sys, 'argv', ['prepare_general', '--root', str(root),
                                        '--legacy-root', str(legacy), '--freeze-only'])
    assert prep.main() == 0
    original_bytes = (root / 'dataset/split.json').read_bytes()
    original_hash = prep.sha256(root / 'dataset/split.json')
    archived = json.loads(original_bytes)

    relocated = tmp_path / 'relocated'
    original.rename(relocated)
    root = relocated / 'general'
    legacy = relocated / 'legacy'
    monkeypatch.chdir(tmp_path)
    legacy_arg = legacy.relative_to(tmp_path) if relative_legacy else legacy
    received = []

    def prepare(frozen_root, coco, split, **kwargs):
        assert frozen_root == root
        assert coco == source
        assert split == archived == json.loads((root / 'dataset/split.json').read_bytes())
        assert kwargs['legacy_root'] == legacy_arg
        assert kwargs['legacy_samples'] == {row['id']: row for row in old['samples']}
        received.append(split)
        return {'complete': False, 'verified_images': 0, 'new_https_downloads_this_run': 0,
                'legacy_cache_copies_this_run': 0, 'failures': []}

    # Spy only at the preparation boundary: freezing and CLI loading run for real.
    monkeypatch.setattr(prep, 'prepare_frozen', prepare)
    monkeypatch.setattr(prep.sys, 'argv', ['prepare_general', '--root', str(root),
                                        '--legacy-root', str(legacy_arg)])
    assert prep.main() == 1
    assert received == [archived]
    assert (root / 'dataset/split.json').read_bytes() == original_bytes
    assert prep.sha256(root / 'dataset/split.json') == original_hash


def test_split_is_deterministic_disjoint_and_covers_all_categories(coco, dev_ids):
    prep = module()
    split = prep.make_split(coco, dev_ids)
    reordered = {**coco, 'images': list(reversed(coco['images'])),
                 'annotations': list(reversed(coco['annotations'])),
                 'categories': list(reversed(coco['categories']))}
    assert split == prep.make_split(reordered, list(reversed(dev_ids)))
    dev, holdout = set(split['dev_ids']), set(split['acceptance_ids'])
    profile = split['profile_samples']
    selected = {s['id'] for s in profile}
    assert len(dev) == 40 and len(holdout) == 4960
    assert not dev & holdout
    assert dev | holdout == {im['id'] for im in coco['images']}
    assert len(profile) == len(selected) == 80
    assert selected <= holdout
    assert Counter(s['stratum_name'] for s in profile) == {
        'person': 10, 'traffic_vehicles': 10, 'animals': 10, 'sports': 10,
        'food_kitchen': 10, 'furniture': 10, 'electronics': 10, 'other_objects': 10}
    groups = split['strata']
    all_categories = [cid for group in groups for cid in group['category_ids']]
    assert len(all_categories) == len(set(all_categories)) == 80
    assert set(all_categories) == {c['id'] for c in coco['categories']}
    assert {a['category_id'] for a in coco['annotations'] if a['image_id'] in holdout} == set(all_categories)
    group_ids = {g['name']: set(g['category_ids']) for g in groups}
    for row in profile:
        assert any(a['image_id'] == row['id'] and not a['iscrowd']
                   and a['category_id'] in group_ids[row['stratum_name']]
                   and a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in coco['annotations'])
    licenses = {im['license'] for im in coco['images'] if im['id'] in selected}
    assert len(licenses) > 1
    assert split['seed'] == 20261106


def test_frozen_split_cannot_be_replaced(tmp_path, coco, dev_ids):
    prep = module()
    assert hasattr(prep, 'freeze_split'), 'immutable split freezing is not implemented'
    split = prep.make_split(coco, dev_ids)
    path = tmp_path / 'dataset/split.json'
    prep.freeze_split(path, split)
    frozen_bytes = path.read_bytes()
    prep.freeze_split(path, split)
    assert path.read_bytes() == frozen_bytes
    changed = {**split, 'seed': 20261107}
    with pytest.raises(ValueError, match='frozen'):
        prep.freeze_split(path, changed)
    assert path.read_bytes() == frozen_bytes


def test_download_keeps_failed_id_and_resumes_with_all_ground_truth(tmp_path):
    import cv2
    import hashlib
    import numpy as np
    prep = module()
    assert hasattr(prep, 'prepare_frozen'), 'frozen download and audit are not implemented'
    images = [{'id': n, 'width': 40, 'height': 30, 'license': n - 10,
               'file_name': f'{n:012d}.jpg', 'flickr_url': f'http://example.org/{n}',
               'coco_url': f'http://images.cocodataset.org/val2017/{n:012d}.jpg'}
              for n in (11, 12, 13)]
    annotations = [{'id': 100 + n, 'image_id': n, 'category_id': n - 10,
                    'bbox': [1, 2, 10, 20], 'area': 200, 'iscrowd': 0,
                    'segmentation': [[1, 2, 11, 2, 11, 22, 1, 22]]} for n in (11, 12, 13)]
    annotations.append({'id': 999, 'image_id': 12, 'category_id': 1,
                        'bbox': [0, 0, 40, 30], 'area': 1200, 'iscrowd': 1,
                        'segmentation': {'counts': '0', 'size': [30, 40]}})
    source = {'info': {'fixture': True}, 'images': images, 'annotations': annotations,
              'licenses': [{'id': n, 'name': f'license{n}', 'url': f'http://license.org/{n}'} for n in (1, 2, 3)],
              'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'},
                             {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'},
                             {'id': 3, 'name': 'car', 'supercategory': 'vehicle'}]}
    split = {'dev_ids': [11], 'acceptance_ids': [12, 13],
             'profile_samples': [{'id': 12, 'stratum_name': 'traffic_vehicles'}],
             'strata': [], 'seed': 20261106, 'per_stratum': 1}
    prep.freeze_split(tmp_path / 'dataset/split.json', split)
    frozen_bytes = (tmp_path / 'dataset/split.json').read_bytes()
    calls = []
    fail = {13}
    payloads = {n: cv2.imencode('.jpg', np.full((30, 40, 3), n, np.uint8))[1].tobytes()
                for n in (11, 12, 13)}

    def fetch(image):
        assert (tmp_path / 'dataset/split.json').read_bytes() == frozen_bytes
        calls.append(image['id'])
        if image['id'] in fail:
            raise OSError('deliberate fixture download failure')
        return payloads[image['id']]

    audit = prep.prepare_frozen(tmp_path, source, split, workers=2, attempts=1, downloader=fetch)
    assert sorted(calls) == [11, 12, 13]
    assert audit['verified_images'] == 2
    assert [e['id'] for e in audit['failures']] == [13]
    holdout = json.loads((tmp_path / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
    assert [s['id'] for s in holdout['samples']] == [12, 13]
    assert holdout['samples'][1]['status'] == 'error'
    subset = json.loads((tmp_path / 'dataset/instances_profile80.json').read_text(encoding='utf-8'))
    assert subset['annotations'] == [a for a in annotations if a['image_id'] == 12]
    assert subset['categories'] == source['categories']
    assert subset['licenses'] == source['licenses']
    profile = json.loads((tmp_path / 'dataset/samples.json').read_text(encoding='utf-8'))
    row = profile['samples'][0]
    assert row['annotation_ids'] == [112, 999]
    assert row['category_ids'] == [1, 2]
    assert row['noncrowd_instance_count'] == 1 and row['crowd_instance_count'] == 1
    assert row['scale_counts_noncrowd'] == {'small': 1, 'medium': 0, 'large': 0}
    assert row['license_info'] == source['licenses'][1]
    assert row['image_sha256'] == hashlib.sha256(payloads[12]).hexdigest()
    assert hashlib.sha256((tmp_path / row['input']).read_bytes()).hexdigest() == row['input_sha256']
    assert row['input_shape'] == [640, 640, 3]
    assert row['download_url'] == 'https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000012.jpg'
    assert row['scene_review'] == '未审图' and row['scene_label_source'] is None
    assert len(list((tmp_path / 'dataset/inputs').glob('*.u8'))) == 1
    calls.clear()
    fail.clear()
    audit = prep.prepare_frozen(tmp_path, source, split, workers=2, attempts=1, downloader=fetch)
    assert calls == [13]
    assert audit['verified_images'] == 3 and audit['failures'] == []
    assert audit['complete'] is True
    assert (tmp_path / 'dataset/split.json').read_bytes() == frozen_bytes
    assert len((tmp_path / 'dataset/download_progress.jsonl').read_text(encoding='utf-8').splitlines()) >= 6


@pytest.fixture
def synthetic_development(tmp_path):
    """One synthetic frozen development image; no real COCO files or network."""
    import hashlib
    from types import SimpleNamespace
    import cv2
    import numpy as np

    prep = module()
    root = tmp_path / 'general'
    legacy_root = tmp_path / 'legacy'
    image = {'id': 11, 'width': 40, 'height': 30, 'license': 1,
             'file_name': '000000000011.jpg'}
    coco = {'images': [image], 'annotations': [],
            'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}],
            'licenses': [{'id': 1, 'name': 'synthetic license'}]}
    split = {'dev_ids': [11], 'acceptance_ids': [], 'profile_samples': [],
             'strata': [], 'seed': 20261106, 'per_stratum': 1}
    frozen_path = root / 'dataset/split.json'
    prep.freeze_split(frozen_path, split)
    encoded, buffer = cv2.imencode('.jpg', np.full((30, 40, 3), (17, 63, 209), np.uint8))
    assert encoded
    payload = buffer.tobytes()
    rgb, _ = prep.decode_input(payload, image)
    old = {'id': 11, 'image': f'dataset/images/{image["file_name"]}',
           'image_sha256': hashlib.sha256(payload).hexdigest(),
           'input_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
    return SimpleNamespace(prep=prep, root=root, legacy_root=legacy_root, coco=coco,
                           image=image, split=split, old=old, payload=payload,
                           frozen_path=frozen_path, frozen_bytes=frozen_path.read_bytes(),
                           frozen_hash=prep.sha256(frozen_path))


@pytest.mark.parametrize('previous_failure', [False, True])
def test_missing_legacy_cache_downloads_same_frozen_id(synthetic_development, previous_failure):
    case = synthetic_development
    if previous_failure:
        journal = case.root / 'dataset/download_progress.jsonl'
        failed = {'id': 11, 'status': 'error', 'image_sha256': None, 'input_sha256': None,
                  'error': 'OSError: previous download failed'}
        journal.write_text(json.dumps(failed) + '\n', encoding='utf-8')
        assert case.prep.read_journal(journal) == {}
    calls = []

    def fetch(image):
        assert image == case.image
        calls.append(image['id'])
        assert case.frozen_path.read_bytes() == case.frozen_bytes
        return case.payload

    audit = case.prep.prepare_frozen(case.root, case.coco, case.split,
                                    legacy_root=case.legacy_root, legacy_samples={11: case.old},
                                    workers=1, attempts=1, downloader=fetch)

    assert audit['complete'] and audit['verified_images'] == 1
    assert calls == [11]
    assert audit['new_https_downloads_this_run'] == 1
    assert audit['legacy_cache_copies_this_run'] == 0
    row = json.loads((case.root / 'dataset/dev_samples.json').read_bytes())['samples'][0]
    assert row['id'] == 11 and row['status'] == 'ok'
    assert row['acquisition_method'] == 'https_download'
    assert row['image_sha256'] == case.old['image_sha256']
    assert row['input_sha256'] == case.old['input_sha256']
    assert (case.root / row['image']).read_bytes() == case.payload
    assert not (case.legacy_root / case.old['image']).exists()
    assert case.frozen_path.read_bytes() == case.frozen_bytes
    assert case.prep.sha256(case.frozen_path) == case.frozen_hash


@pytest.mark.parametrize('cache_location', ['legacy', 'current'])
def test_corrupt_development_cache_fails_without_download(synthetic_development, cache_location):
    case = synthetic_development
    cache_root = case.legacy_root if cache_location == 'legacy' else case.root
    cache = cache_root / case.old['image']
    cache.parent.mkdir(parents=True, exist_ok=True)
    corrupt = b'deliberately corrupt synthetic JPEG'
    cache.write_bytes(corrupt)
    calls = []

    def fetch(image):
        calls.append(image['id'])
        return case.payload

    audit = case.prep.prepare_frozen(case.root, case.coco, case.split,
                                    legacy_root=case.legacy_root, legacy_samples={11: case.old},
                                    workers=1, attempts=1, downloader=fetch)

    assert not audit['complete'] and audit['verified_images'] == 0
    assert [row['id'] for row in audit['failures']] == [11]
    assert 'image SHA mismatch' in audit['failures'][0]['error']
    assert calls == []
    assert cache.read_bytes() == corrupt
    assert case.frozen_path.read_bytes() == case.frozen_bytes
    assert case.prep.sha256(case.frozen_path) == case.frozen_hash


@pytest.mark.parametrize(('hash_field', 'message'), [
    ('image_sha256', 'image SHA mismatch'),
    ('input_sha256', 'RGB input SHA mismatch'),
])
@pytest.mark.parametrize('previous_status', [None, 'error', 'ok'])
def test_download_enforces_frozen_development_hashes(synthetic_development, hash_field,
                                                      message, previous_status):
    case = synthetic_development
    # The frozen manifest disagrees with these bytes; a later journal must not authorize them.
    expected = {**case.old, hash_field: '0' * 64}
    if previous_status is not None:
        previous = {**case.old, 'status': previous_status}
        if previous_status == 'error':
            previous.update(image_sha256=None, input_sha256=None)
        (case.root / 'dataset/download_progress.jsonl').write_text(
            json.dumps(previous) + '\n', encoding='utf-8')
    calls = []

    def fetch(image):
        calls.append(image['id'])
        return case.payload

    audit = case.prep.prepare_frozen(case.root, case.coco, case.split,
                                    legacy_root=case.legacy_root, legacy_samples={11: expected},
                                    workers=1, attempts=1, downloader=fetch)

    assert not audit['complete'] and audit['verified_images'] == 0
    assert calls == [11]
    assert [row['id'] for row in audit['failures']] == [11]
    assert message in audit['failures'][0]['error']
    assert not (case.root / case.old['image']).exists()
    assert case.frozen_path.read_bytes() == case.frozen_bytes
    assert case.prep.sha256(case.frozen_path) == case.frozen_hash


@pytest.mark.parametrize('previous_hash', [None, '0' * 64])
def test_development_source_hashes_override_stale_resume(synthetic_development, previous_hash):
    case = synthetic_development
    previous = {'id': 11, 'status': 'ok',
                'image_sha256': previous_hash, 'input_sha256': previous_hash}
    (case.root / 'dataset/download_progress.jsonl').write_text(
        json.dumps(previous) + '\n', encoding='utf-8')
    calls = []

    def fetch(image):
        calls.append(image['id'])
        return case.payload

    audit = case.prep.prepare_frozen(case.root, case.coco, case.split,
                                    legacy_root=case.legacy_root, legacy_samples={11: case.old},
                                    workers=1, attempts=1, downloader=fetch)

    assert audit['complete'] and audit['verified_images'] == 1
    assert calls == [11]
    row = json.loads((case.root / 'dataset/dev_samples.json').read_bytes())['samples'][0]
    assert row['image_sha256'] == case.old['image_sha256']
    assert row['input_sha256'] == case.old['input_sha256']
    assert case.frozen_path.read_bytes() == case.frozen_bytes
    assert case.prep.sha256(case.frozen_path) == case.frozen_hash


def test_resume_recovers_only_an_interrupted_final_jsonl_record(tmp_path):
    prep = module()
    journal = tmp_path / 'download_progress.jsonl'
    valid = b'{"id": 1, "status": "ok"}\n{"id": 2, "status": "ok"}\n'
    interrupted = b'{"id": 3, "status": '
    journal.write_bytes(valid + interrupted)
    assert prep.read_journal(journal) == {1: {'id': 1, 'status': 'ok'}, 2: {'id': 2, 'status': 'ok'}}
    assert journal.read_bytes() == valid
    assert [p.read_bytes() for p in tmp_path.glob('*.interrupted_tail.*')] == [interrupted]
    journal.write_bytes(b'bad middle line\n' + valid)
    with pytest.raises(ValueError):
        prep.read_journal(journal)


@pytest.mark.skipif(not (ROOT / 'evidence/data_audit.json').exists(), reason='Run data preparation before artifact verification')
def test_materialized_data_matches_every_frozen_id_hash_and_complete_gt(coco, dev_ids):
    import hashlib
    import cv2
    import numpy as np
    from collections import defaultdict
    prep = module()
    dataset = ROOT / 'dataset'
    audit = json.loads((ROOT / 'evidence/data_audit.json').read_text(encoding='utf-8'))
    assert audit['complete'] and audit['verified_images'] == 5000 and not audit['failures']
    annotation_path = dataset / 'instances_val2017.json'
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == prep.ANNOTATION_SHA256
    split = json.loads((dataset / 'split.json').read_text(encoding='utf-8'))
    assert split['dev_ids'] == sorted(dev_ids)
    manifests = {name: json.loads((dataset / name).read_text(encoding='utf-8'))
                 for name in ('dev_samples.json', 'acceptance_samples.json', 'samples.json')}
    assert {name: m['count'] for name, m in manifests.items()} == {
        'dev_samples.json': 40, 'acceptance_samples.json': 4960, 'samples.json': 80}
    source_annotations = defaultdict(list)
    source_images = {im['id']: im for im in coco['images']}
    for annotation in coco['annotations']:
        source_annotations[annotation['image_id']].append(annotation)
    verified_files = set()
    for name, manifest in manifests.items():
        ids = [row['id'] for row in manifest['samples']]
        expected = (split['dev_ids'] if name == 'dev_samples.json' else
                    split['acceptance_ids'] if name == 'acceptance_samples.json' else
                    [row['id'] for row in split['profile_samples']])
        assert ids == expected and len(ids) == len(set(ids))
        selected = set(ids)
        subset = json.loads((ROOT / manifest['subset_annotation']).read_text(encoding='utf-8'))
        assert subset['annotations'] == [a for a in coco['annotations'] if a['image_id'] in selected]
        assert subset['images'] == [im for im in coco['images'] if im['id'] in selected]
        assert subset['categories'] == coco['categories'] and subset['licenses'] == coco['licenses']
        for row in manifest['samples']:
            assert row['status'] == 'ok'
            assert row['annotation_ids'] == [a['id'] for a in source_annotations[row['id']]]
            assert row['category_ids'] == sorted({a['category_id'] for a in source_annotations[row['id']]})
            assert all(row[k] == v for k, v in source_images[row['id']].items())
            raw = (ROOT / row['image']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row['image_sha256']
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            assert bgr.shape == (row['height'], row['width'], 3)
            rgb, transform = prep.preprocessing()[0](bgr)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['input_sha256']
            assert transform == row['transform']
            if row['input']:
                assert hashlib.sha256((ROOT / row['input']).read_bytes()).hexdigest() == row['input_sha256']
            verified_files.add(row['image'])
    assert len(verified_files) == 5000
    assert len(list((dataset / 'images').glob('*.jpg'))) == 5000
    assert len(list((dataset / 'inputs').glob('*.u8'))) == 80
    legacy = json.loads((LEGACY / 'dataset/samples.json').read_text(encoding='utf-8'))
    originals = {row['id']: row for row in legacy['samples']}
    for row in manifests['dev_samples.json']['samples']:
        assert row['image_sha256'] == originals[row['id']]['image_sha256']
        assert row['input_sha256'] == originals[row['id']]['input_sha256']
    for relative, expected in audit['artifacts'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected['sha256']
