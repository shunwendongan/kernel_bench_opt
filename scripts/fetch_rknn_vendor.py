#!/usr/bin/env python3
"""Fetch pinned RKNN files directly from their original vendor into a local folder."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/local/runtime/vendor-rknn')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets/rknn-vendor-source.json').read_text())
    for item in manifest['files']:
        target = args.output / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != item['sha256']:
                raise SystemExit(f'Existing file differs: {target}; choose another output directory')
        else:
            temporary = target.with_name(target.name + '.part')
            with urllib.request.urlopen(item['url'], timeout=60) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if digest(temporary) != item['sha256']:
                raise SystemExit(f'Vendor SHA-256 mismatch: {item["path"]}')
            temporary.replace(target)
        print(f'verified: {item["path"]}')
    print('Vendor files ready. Existing system files and drivers were not changed.')


if __name__ == '__main__':
    main()
