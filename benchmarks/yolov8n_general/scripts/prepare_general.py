"""Freeze and verify general COCO val2017 data; never substitute failures."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import types

import cv2
import numpy as np
from PIL import Image
import PIL
import requests


def freeze_split(path, split):
    """Create once; reuse archived content when only its source location changes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8', newline='\n') as output:
            json.dump(split, output, ensure_ascii=False, indent=2)
            output.write('\n')
    except FileExistsError:
        frozen = json.loads(path.read_text(encoding='utf-8'))
        # The source path locates a manifest; its hash freezes the source identity.
        frozen_identity = {key: value for key, value in frozen.items() if key != 'development_source'}
        proposed_identity = {key: value for key, value in split.items() if key != 'development_source'}
        if frozen_identity != proposed_identity:
            raise ValueError('Existing frozen split differs; refusing to replace IDs')
        return frozen
    return split

ANNOTATION_SHA256 = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
STRATUM_SUPERCATEGORIES = (
    ('person', ('person',)),
    ('traffic_vehicles', ('vehicle',)),
    ('animals', ('animal',)),
    ('sports', ('sports',)),
    ('food_kitchen', ('food', 'kitchen')),
    ('furniture', ('furniture',)),
    ('electronics', ('electronic',)),
    ('other_objects', ('outdoor', 'accessory', 'appliance', 'indoor')),
)


def make_split(coco, dev_ids, seed=20261106, per_stratum=10):
    image_ids = {im['id'] for im in coco['images']}
    if len(image_ids) != len(coco['images']):
        raise ValueError('Duplicate COCO image IDs')
    if len(set(dev_ids)) != len(dev_ids) or not set(dev_ids) <= image_ids:
        raise ValueError('Development IDs must be unique COCO image IDs')
    categories = sorted(coco['categories'], key=lambda c: c['id'])
    strata = [{'name': name, 'source_supercategories': list(supers),
               'category_ids': [c['id'] for c in categories if c['supercategory'] in supers],
               'category_names': [c['name'] for c in categories if c['supercategory'] in supers]}
              for name, supers in STRATUM_SUPERCATEGORIES]
    grouped = [cid for group in strata for cid in group['category_ids']]
    if len(grouped) != len(set(grouped)) or set(grouped) != {c['id'] for c in categories}:
        raise ValueError('Strata must partition every source category exactly once')
    used = set(dev_ids)
    rng = random.Random(seed)
    profile = []
    for group in strata:
        eligible = sorted({a['image_id'] for a in coco['annotations']
                           if a['category_id'] in group['category_ids']
                           and not a['iscrowd'] and a['bbox'][2] > 0 and a['bbox'][3] > 0} - used)
        for image_id in rng.sample(eligible, per_stratum):
            profile.append({'id': image_id, 'stratum_name': group['name']})
            used.add(image_id)
    return {'schema_version': 1, 'dataset': 'COCO val2017', 'seed': seed,
            'per_stratum': per_stratum, 'strata': strata,
            'dev_ids': sorted(dev_ids), 'acceptance_ids': sorted(image_ids - set(dev_ids)),
            'profile_samples': profile,
            'sampling_definition': 'Strata in declared order; sorted eligible IDs; Random(seed).sample; exclude development and previous selections; non-crowd positive-width/height boxes; no license filter.',
            'failure_policy': 'Keep frozen IDs and report errors; never substitute another image.',
            'scene_scope': 'COCO category strata; no claim of manually verified scene labels.'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    with temporary.open('wb') as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


@lru_cache(maxsize=1)
def preprocessing():
    source = Path(__file__).resolve().with_name('yolo_common.py')
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / 'yolov8n_home/scripts/yolo_common.py'
    payload = source.read_bytes()
    module = types.ModuleType('general_data_letterbox')
    exec(compile(payload, str(source), 'exec'), module.__dict__)
    return module.letterbox, {'source': str(source), 'source_sha256': hashlib.sha256(payload).hexdigest(),
                            'decode': 'OpenCV IMREAD_COLOR (BGR uint8), strict Pillow load also checked',
                            'resize': 'OpenCV INTER_LINEAR, aspect-ratio preserving',
                            'pad': 114, 'shape': [640, 640, 3], 'layout': 'HWC', 'dtype': 'uint8',
                            'color': 'RGB', 'normalization': 'none'}


def decode_input(raw, image):
    with Image.open(io.BytesIO(raw)) as decoded:
        decoded.load()
        if decoded.size != (image['width'], image['height']):
            raise ValueError(f'Pillow dimension mismatch for {image["id"]}')
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (image['height'], image['width'], 3):
        raise ValueError(f'OpenCV decode or dimension mismatch for {image["id"]}')
    return preprocessing()[0](bgr)


def annotation_metadata(annotations):
    noncrowd = [a for a in annotations if not a['iscrowd']]
    scales = {'small': 0, 'medium': 0, 'large': 0}
    for annotation in noncrowd:
        area = annotation['area']
        scales['small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large'] += 1
    return {'annotation_ids': [a['id'] for a in annotations],
            'category_ids': sorted({a['category_id'] for a in annotations}),
            'noncrowd_instance_count': len(noncrowd),
            'crowd_instance_count': len(annotations) - len(noncrowd),
            'scale_counts_noncrowd': scales,
            'instance_scales': [{'annotation_id': a['id'], 'category_id': a['category_id'],
                                 'iscrowd': a['iscrowd'], 'area': a['area'],
                                 'bbox_width': a['bbox'][2], 'bbox_height': a['bbox'][3],
                                 'area_fraction': None} for a in annotations]}


_thread_session = threading.local()


def download_image(image):
    if not hasattr(_thread_session, 'session'):
        session = requests.Session()
        session.trust_env = False  # Do not consult netrc credentials or environment proxies.
        _thread_session.session = session
    url = f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{image["file_name"]}'
    with _thread_session.session.get(url, timeout=(15, 90)) as response:
        response.raise_for_status()
        return response.content


def read_journal(path):
    """Preserve an interrupted tail separately, never skip corrupt middle rows."""
    path = Path(path)
    records = {}
    if not path.exists():
        return records
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        if line.strip():
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b'\n'):
                    raise ValueError(f'Corrupt journal record {index + 1}')
                atomic_bytes(path.with_name(f'{path.name}.interrupted_tail.{time.time_ns()}'), line)
                atomic_bytes(path, payload[:offset])
                break
            if row.get('status') == 'ok':
                records[row['id']] = row
        offset += len(line)
    else:
        if payload and not payload.endswith(b'\n'):
            atomic_bytes(path, payload + b'\n')
    return records


def prepare_frozen(root, coco, split, *, legacy_root=None, legacy_samples=None,
                   workers=12, attempts=5, downloader=None):
    """Process every frozen ID, then read/decode/hash every saved image again."""
    root = Path(root)
    dataset = root / 'dataset'
    split_path = dataset / 'split.json'
    if not split_path.is_file() or json.loads(split_path.read_text(encoding='utf-8')) != split:
        raise ValueError('Must write the exact frozen split before any downloads')
    split_hash = sha256(split_path)
    started = utc_now()
    images = {im['id']: im for im in coco['images']}
    all_ids = sorted(split['dev_ids'] + split['acceptance_ids'])
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(images):
        raise ValueError('Frozen dev and acceptance IDs must partition every source image')
    profile = {s['id']: s['stratum_name'] for s in split['profile_samples']}
    if len(profile) != len(split['profile_samples']) or not set(profile) <= set(split['acceptance_ids']):
        raise ValueError('Profile IDs must be unique and excluded from development')
    annotations = defaultdict(list)
    for annotation in coco['annotations']:
        annotations[annotation['image_id']].append(annotation)
    licenses = {item['id']: item for item in coco['licenses']}
    source_rows = {im['id']: index for index, im in enumerate(coco['images'])}
    legacy_samples = legacy_samples or {}
    previous = read_journal(dataset / 'download_progress.jsonl')
    fetch = downloader or download_image
    (dataset / 'images').mkdir(parents=True, exist_ok=True)
    (dataset / 'inputs').mkdir(exist_ok=True)
    cv2.setNumThreads(1)

    def process(image_id):
        image = images[image_id]
        path = dataset / 'images' / image['file_name']
        base = {**image, 'sample_id': f'coco_val2017_{image_id:012d}',
                'source_image_row': source_rows[image_id],
                'stratum_name': profile.get(image_id),
                'image': path.relative_to(root).as_posix(),
                'download_url': f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{image["file_name"]}',
                'license_info': licenses[image['license']],
                'source': 'COCO val2017 original image and instances_val2017 metadata',
                'scene_review': '未审图', 'scene_label_source': None,
                **annotation_metadata(annotations[image_id])}
        for item in base['instance_scales']:
            item['area_fraction'] = item['area'] / (image['width'] * image['height'])
        # Frozen development provenance takes precedence over mutable resume records.
        expected = legacy_samples.get(image_id) or previous.get(image_id)
        attempt_errors = []
        for attempt in range(attempts):
            try:
                if path.exists():
                    raw, method = path.read_bytes(), 'existing_verified_cache'
                elif image_id in legacy_samples and legacy_root is not None:
                    old = legacy_samples[image_id]
                    try:
                        raw = (Path(legacy_root) / old['image']).read_bytes()
                    except FileNotFoundError:
                        raw, method = fetch(image), 'https_download'
                    else:
                        method = 'legacy_cache_copy'
                else:
                    raw, method = fetch(image), 'https_download'
                image_hash = hashlib.sha256(raw).hexdigest()
                if expected and expected['image_sha256'] != image_hash:
                    raise ValueError(f'Cached/source image SHA mismatch for {image_id}')
                rgb, transform = decode_input(raw, image)
                input_bytes = rgb.tobytes()
                input_hash = hashlib.sha256(input_bytes).hexdigest()
                if expected and expected['input_sha256'] != input_hash:
                    raise ValueError(f'Reconstructed RGB input SHA mismatch for {image_id}')
                if not path.exists():
                    atomic_bytes(path, raw)
                input_path = None
                if image_id in profile:
                    target = dataset / 'inputs' / f'{image_id:012d}.u8'
                    atomic_bytes(target, input_bytes)
                    input_path = target.relative_to(root).as_posix()
                return {**base, 'status': 'ok', 'error': None, 'image_sha256': image_hash,
                        'image_bytes': len(raw), 'input': input_path, 'input_sha256': input_hash,
                        'input_shape': list(rgb.shape), 'input_dtype': str(rgb.dtype),
                        'input_layout': 'HWC', 'input_color': 'RGB', 'transform': transform,
                        'acquisition_method': method, 'attempt_errors': attempt_errors,
                        'verified_at_utc': utc_now(), 'split_sha256': split_hash}
            except Exception as error:
                attempt_errors.append({'attempt': attempt + 1, 'error': f'{type(error).__name__}: {error}'})
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 16))
        return {**base, 'status': 'error', 'error': attempt_errors[-1]['error'],
                'image_sha256': None, 'input': None, 'input_sha256': None,
                'attempt_errors': attempt_errors, 'split_sha256': split_hash}

    results = {}
    with (dataset / 'download_progress.jsonl').open('a', encoding='utf-8', newline='\n') as journal:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, image_id): image_id for image_id in all_ids}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                results[row['id']] = row
                journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                if index % 100 == 0 or index == len(all_ids):
                    print(json.dumps({'phase': 'download', 'completed': index, 'total': len(all_ids),
                                      'failures': sum(r['status'] != 'ok' for r in results.values()),
                                      'utc': utc_now()}), flush=True)

    # Separate full readback: detect changed files, partial writes and input mismatches.
    verified = []
    failures = []
    for index, image_id in enumerate(all_ids, 1):
        row = results[image_id]
        try:
            if row['status'] != 'ok':
                raise ValueError(row['error'])
            raw = (root / row['image']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row['image_sha256']:
                raise ValueError('Original image SHA changed during final readback')
            rgb, transform = decode_input(raw, images[image_id])
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row['input_sha256'] or transform != row['transform']:
                raise ValueError('RGB input hash or transform changed during final readback')
            if row['input'] and sha256(root / row['input']) != row['input_sha256']:
                raise ValueError('Saved profile tensor hash mismatch')
            if row['annotation_ids'] != [a['id'] for a in annotations[image_id]]:
                raise ValueError('Incomplete annotation IDs')
            verified.append(image_id)
        except Exception as error:
            row['status'] = 'error'
            row['error'] = f'{type(error).__name__}: {error}'
            failures.append({'id': image_id, 'error': row['error'], 'attempt_errors': row['attempt_errors']})
        if index % 500 == 0 or index == len(all_ids):
            print(json.dumps({'phase': 'readback', 'completed': index, 'total': len(all_ids),
                              'failures': len(failures), 'utc': utc_now()}), flush=True)

    annotation_path = dataset / 'instances_val2017.json'
    if not annotation_path.exists():
        write_json(annotation_path, coco)
    common = {'dataset': 'COCO val2017', 'seed': split['seed'],
              'annotation_source': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
              'annotation_member': 'annotations/instances_val2017.json',
              'annotation_sha256': sha256(annotation_path),
              'frozen_split': 'dataset/split.json', 'split_sha256': split_hash,
              'categories': coco['categories'], 'licenses': coco['licenses'],
              'preprocessing': preprocessing()[1],
              'scale_definition': 'Non-crowd segmentation area: small <1024; medium >=1024 and <9216; large >=9216. Disjoint reporting bins; COCOeval uses its own standard ranges.',
              'scene_scope': 'Category labels only; no manual scene verification.'}
    manifests = {}
    for name, ids, subset_name in (
        ('dev_samples.json', split['dev_ids'], 'instances_dev40.json'),
        ('acceptance_samples.json', split['acceptance_ids'], 'instances_acceptance.json'),
        ('samples.json', [s['id'] for s in split['profile_samples']], 'instances_profile80.json'),
    ):
        selected = set(ids)
        subset = {**coco, 'images': [im for im in coco['images'] if im['id'] in selected],
                  'annotations': [a for a in coco['annotations'] if a['image_id'] in selected]}
        write_json(dataset / subset_name, subset)
        manifest = {**common, 'count': len(ids), 'subset_annotation': f'dataset/{subset_name}',
                    'subset_annotation_sha256': sha256(dataset / subset_name),
                    'strata': split['strata'], 'per_stratum': split['per_stratum'],
                    'samples': [results[image_id] for image_id in ids]}
        write_json(dataset / name, manifest)
        manifests[name] = {'sha256': sha256(dataset / name), 'count': len(ids),
                           'annotations': len(subset['annotations']),
                           'crowd_annotations': sum(bool(a['iscrowd']) for a in subset['annotations']),
                           'category_count': len({a['category_id'] for a in subset['annotations']}),
                           'zero_annotation_images': sum(not annotations[i] for i in ids)}

    categories = []
    holdout = set(split['acceptance_ids'])
    for category in coco['categories']:
        source = [a for a in coco['annotations'] if a['category_id'] == category['id']]
        held = [a for a in source if a['image_id'] in holdout and not a['iscrowd']]
        categories.append({**category, 'all_annotations': len(source),
                           'acceptance_noncrowd_instances': len(held),
                           'acceptance_noncrowd_images': len({a['image_id'] for a in held})})
    digest_rows = [{'id': i, 'image_sha256': results[i].get('image_sha256'),
                    'input_sha256': results[i].get('input_sha256')} for i in all_ids]
    write_json(dataset / 'image_input_hashes.json', digest_rows)
    artifacts = [annotation_path, split_path, dataset / 'image_input_hashes.json',
                 *[dataset / name for name in manifests],
                 *[dataset / name for name in ('instances_dev40.json', 'instances_acceptance.json', 'instances_profile80.json')]]
    duplicate_hashes = defaultdict(list)
    for image_id in verified:
        duplicate_hashes[results[image_id]['image_sha256']].append(image_id)
    audit = {'schema_version': 1, 'started_at_utc': started, 'finished_at_utc': utc_now(),
             'complete': len(verified) == len(all_ids) and not failures,
             'frozen_images': len(all_ids), 'verified_images': len(verified),
             'dev_count': len(split['dev_ids']), 'acceptance_count': len(split['acceptance_ids']),
             'profile_count': len(profile), 'saved_profile_inputs': sum(bool(results[i].get('input')) for i in verified),
             'new_https_downloads_this_run': sum(results[i]['acquisition_method'] == 'https_download' for i in verified),
             'legacy_cache_copies_this_run': sum(results[i]['acquisition_method'] == 'legacy_cache_copy' for i in verified),
             'existing_cache_verified_this_run': sum(results[i]['acquisition_method'] == 'existing_verified_cache' for i in verified),
             'image_bytes': sum(results[i]['image_bytes'] for i in verified),
             'full_annotation_count': len(coco['annotations']),
             'full_crowd_count': sum(bool(a['iscrowd']) for a in coco['annotations']),
             'full_category_count': len(coco['categories']),
             'profile_stratum_counts': dict(Counter(profile.values())),
             'license_image_counts': dict(Counter(str(images[i]['license']) for i in all_ids)),
             'manifests': manifests, 'category_coverage': categories,
             'duplicate_content_image_ids': [ids for ids in duplicate_hashes.values() if len(ids) > 1],
             'failures': failures,
             'retry_errors': [{'id': i, 'errors': results[i]['attempt_errors']} for i in all_ids if results[i]['attempt_errors']],
             'preprocessing': preprocessing()[1],
             'verification': 'All frozen IDs read back from disk; strict Pillow load; OpenCV decode and exact dimensions; reconstruct RGB and compare SHA256, transform, saved profile tensor and complete annotation IDs.',
             'environment': {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
                             'pillow': PIL.__version__, 'requests': requests.__version__},
             'artifacts': {path.relative_to(root).as_posix(): {'sha256': sha256(path), 'bytes': path.stat().st_size}
                           for path in artifacts},
             'script_sha256': sha256(__file__)}
    if sha256(split_path) != split_hash:
        raise ValueError('Frozen split changed during preparation')
    write_json(root / 'evidence/data_audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--legacy-root', type=Path, default=Path(__file__).resolve().parents[2] / 'yolov8n_home')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    source = args.legacy_root / 'dataset/instances_val2017.json'
    if sha256(source) != ANNOTATION_SHA256:
        raise ValueError('Original COCO annotation SHA256 mismatch')
    coco = json.loads(source.read_text(encoding='utf-8'))
    old_path = args.legacy_root / 'dataset/samples.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    if len(coco['images']) != 5000 or len(old['samples']) != 40:
        raise ValueError('Expected 5000 original COCO images and 40 development images')
    split = make_split(coco, [row['id'] for row in old['samples']])
    split.update({'annotation_sha256': ANNOTATION_SHA256,
                  'development_source': str(old_path), 'development_manifest_sha256': sha256(old_path)})
    split = freeze_split(args.root / 'dataset/split.json', split)
    target = args.root / 'dataset/instances_val2017.json'
    if target.exists():
        if sha256(target) != ANNOTATION_SHA256:
            raise ValueError('New annotation copy SHA256 mismatch')
    else:
        atomic_bytes(target, source.read_bytes())
    print(json.dumps({'phase': 'frozen', 'split_sha256': sha256(args.root / 'dataset/split.json'),
                      'dev': len(split['dev_ids']), 'acceptance': len(split['acceptance_ids']),
                      'profile': len(split['profile_samples']), 'utc': utc_now()}), flush=True)
    if args.freeze_only:
        return 0
    audit = prepare_frozen(args.root, coco, split, legacy_root=args.legacy_root,
                           legacy_samples={row['id']: row for row in old['samples']},
                           workers=args.workers, attempts=args.attempts)
    print(json.dumps({key: audit[key] for key in ('complete', 'verified_images', 'new_https_downloads_this_run',
                                                  'legacy_cache_copies_this_run', 'failures')}, ensure_ascii=False), flush=True)
    return 0 if audit['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
