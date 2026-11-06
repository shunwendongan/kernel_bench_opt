"""分别用合成数据检查抽样契约、用真实 COCO 标注检查固定结果。"""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


@pytest.fixture
def synthetic_coco():
    """仅用于单元测试，不作为真实 COCO 数据或实测证据。"""
    categories = [44, 47, 62, 63, 65, 67, 72, 73]
    # 所有类别共用候选图片，确保跨类别去重也受到检查。
    images = [{'id': n, 'file_name': f'synthetic_{n}.jpg',
               'width': 32, 'height': 24, 'license': 1 if n == 44 else 4}
              for n in range(1, 45)]
    annotations = [{'id': n * 100 + category, 'image_id': n,
                    'category_id': category, 'iscrowd': int(n == 41),
                    'bbox': [0, 0, 0 if n == 42 else 10, 0 if n == 43 else 10]}
                   for n in range(1, 45) for category in categories]
    return {'images': images, 'annotations': annotations,
            'categories': [{'id': cid, 'name': f'synthetic_{cid}'} for cid in categories]}


def test_synthetic_sampling_is_deterministic(synthetic_coco):
    from prepare_data import select_samples
    chosen = select_samples(synthetic_coco)
    assert chosen == select_samples(synthetic_coco)
    reordered = {key: list(reversed(rows)) for key, rows in synthetic_coco.items()}
    assert chosen == select_samples(reordered)
    assert chosen != select_samples(synthetic_coco, seed=43)


def test_synthetic_sampling_keeps_category_quotas_without_duplicates(synthetic_coco):
    from prepare_data import select_samples
    chosen = select_samples(synthetic_coco)
    assert Counter(s['stratum_category_id'] for s in chosen) == {
        44: 5, 47: 5, 62: 5, 63: 5, 65: 5, 67: 5, 72: 5, 73: 5}
    assert len(chosen) == len({s['id'] for s in chosen}) == 40
    # 最后四张图片分别只含 crowd、零宽框、零高框或不符许可的标注。
    assert {s['id'] for s in chosen} == set(range(1, 41))
    assert all(s['license'] == 4 for s in chosen)


def test_sampling_is_repeatable_unique_and_keeps_ground_truth():
    assert importlib.util.find_spec('prepare_data') is not None, '尚未实现抽样'
    from prepare_data import select_samples
    annotation_path = ROOT / 'dataset/instances_val2017.json'
    # 仅缺失时跳过；文件存在时，读取和解析错误必须直接失败。
    if not annotation_path.exists():
        pytest.skip(f'缺少外部 COCO 标注：{annotation_path}')
    coco = json.loads(annotation_path.read_text(encoding='utf-8'))
    chosen = select_samples(coco)
    assert chosen == select_samples(coco)
    assert len(chosen) == len({s['id'] for s in chosen}) == 40
    assert [s['id'] for s in chosen[:5]] == [312192, 66231, 8021, 397133, 172877]
    assert all(s['license'] == 4 for s in chosen)
    for sample in chosen:
        anns = [a for a in coco['annotations'] if a['image_id'] == sample['id']]
        assert any(a['category_id'] == sample['stratum_category_id'] and not a['iscrowd']
                   and a['bbox'][2] > 0 and a['bbox'][3] > 0 for a in anns)
    selected_ids = {s['id'] for s in chosen}
    assert sum(a['image_id'] in selected_ids for a in coco['annotations']) == 436
