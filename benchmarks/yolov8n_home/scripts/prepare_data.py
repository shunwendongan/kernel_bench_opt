"""固定 COCO 抽样，下载原始图片并保存可核验输入。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import cv2
import numpy as np
import requests
from yolo_common import letterbox

CATEGORY_IDS = [44, 47, 62, 63, 65, 67, 72, 73]
ANNOTATION_SHA256 = 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'


def select_samples(coco, seed=42, per_category=5):
    images = {im['id']: im for im in coco['images']}
    rng = random.Random(seed)
    used, chosen = set(), []
    for category in CATEGORY_IDS:
        eligible = sorted({a['image_id'] for a in coco['annotations']
                           if a['category_id'] == category and not a['iscrowd']
                           and a['bbox'][2] > 0 and a['bbox'][3] > 0
                           and images[a['image_id']]['license'] == 4} - used)
        for image_id in rng.sample(eligible, per_category):
            chosen.append({**images[image_id], 'stratum_category_id': category})
            used.add(image_id)
    return chosen


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    dataset = root / 'dataset'
    annotation = dataset / 'instances_val2017.json'
    if sha(annotation) != ANNOTATION_SHA256:
        raise ValueError('原始 COCO 标注哈希不一致')
    coco = json.loads(annotation.read_text(encoding='utf-8'))
    selected = select_samples(coco)
    names = {c['id']: c['name'] for c in coco['categories']}
    rows = {im['id']: i for i, im in enumerate(coco['images'])}
    for name in ('images', 'inputs'):
        (dataset / name).mkdir(exist_ok=True)

    def fetch(image):
        image_id = image['id']
        path = dataset / 'images' / image['file_name']
        url = f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{image["file_name"]}'
        if not path.exists():
            response = requests.get(url, timeout=(15, 90))
            response.raise_for_status()
            path.write_bytes(response.content)
        bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (image['height'], image['width']):
            raise ValueError(f'图片尺寸或解码失败: {image_id}')
        rgb, transform = letterbox(bgr)
        input_path = dataset / 'inputs' / f'{image_id:012d}.u8'
        input_path.write_bytes(rgb.tobytes())
        ann_ids = [a['id'] for a in coco['annotations'] if a['image_id'] == image_id]
        return {**image, 'sample_id': f'coco_val2017_{image_id:012d}',
                'source_image_row': rows[image_id], 'stratum_name': names[image['stratum_category_id']],
                'image': path.relative_to(root).as_posix(), 'image_sha256': sha(path),
                'image_bytes': path.stat().st_size, 'download_url': url,
                'input': input_path.relative_to(root).as_posix(), 'input_sha256': sha(input_path),
                'input_shape': list(rgb.shape), 'transform': transform,
                'annotation_ids': ann_ids, 'scene_review': '未审图',
                'scene_label_source': None}

    with ThreadPoolExecutor(max_workers=6) as pool:
        samples = list(pool.map(fetch, selected))
    ids = {s['id'] for s in samples}
    subset = {**coco, 'images': [im for im in coco['images'] if im['id'] in ids],
              'annotations': [a for a in coco['annotations'] if a['image_id'] in ids]}
    subset_path = dataset / 'instances_home40.json'
    subset_path.write_text(json.dumps(subset, ensure_ascii=False), encoding='utf-8')
    result = {'dataset': 'COCO val2017', 'seed': 42, 'count': len(samples),
              'category_order': CATEGORY_IDS, 'per_category': 5,
              'sampling_definition': '按类别顺序、ID升序候选、排除已选ID后 Random(42).sample；非crowd且框宽高>0；license=4',
              'annotation_source': 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
              'annotation_member': 'annotations/instances_val2017.json',
              'annotation_sha256': sha(annotation), 'subset_annotation': 'dataset/instances_home40.json',
              'subset_annotation_sha256': sha(subset_path),
              'scene_scope': '家庭相关物品分层；并非真实家庭域标注',
              'licenses': coco['licenses'], 'categories': coco['categories'], 'samples': samples}
    (dataset / 'samples.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(samples), 'unique_ids': len(ids),
                      'annotations': len(subset['annotations']),
                      'image_bytes': sum(s['image_bytes'] for s in samples),
                      'samples_sha256': sha(dataset / 'samples.json')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    prepare(ap.parse_args().root)
