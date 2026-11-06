#!/usr/bin/env python3
"""Publish a checksum-verified fixed model snapshot to an existing draft Release.

Intended for the manually dispatched GitHub workflow. Never loads a model or
runs inference. HTTP failure messages omit response URLs and signed redirects.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit

from fetch_assets import download_part, member_path, sha256, simple_name
from package_release import package

REPO = Path(__file__).resolve().parents[1]


def require_draft(repository, tag):
    result = subprocess.run(["gh", "release", "view", tag, "--repo", repository,
                             "--json", "isDraft,tagName"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect target Release (gh exit {result.returncode})")
    release = json.loads(result.stdout)
    if not release["isDraft"] or release["tagName"] != tag:
        raise ValueError("Target must be the named draft Release; published assets are not overwritten")


def download_with_retries(item, target):
    for attempt in range(1, 4):
        try:
            return download_part(item, target)
        except Exception as error:
            kind = type(error).__name__
            detail = f"HTTP {error.code}" if isinstance(error, HTTPError) else kind
            print(f"download failed: {item['name']} attempt={attempt}/3 type={detail}", flush=True)
            partial = target.with_name(target.name + ".part")
            if partial.is_file() and partial.stat().st_size >= item["size_bytes"]:
                partial.unlink()
            if attempt == 3:
                raise RuntimeError(f"Download failed for {item['name']} ({detail})") from None
            time.sleep(attempt * 2)


def publish(source_manifest, repository, release_tag, work):
    simple_name(release_tag)
    if len(repository.split("/")) != 2:
        raise ValueError("repository must be owner/repo")
    for component in repository.split("/"):
        simple_name(component)
    manifest = json.loads(source_manifest.read_text())
    model_repository, revision = manifest["repository"], manifest["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a fixed 40-character commit")
    expected_prefix = f"/{model_repository}/resolve/{revision}/"
    files = manifest["files"]
    names = []
    for item in files:
        name = member_path(item["name"]).as_posix()
        if name != item["name"]:
            raise ValueError("Model filenames must be canonical relative POSIX paths")
        url = urlsplit(item["url"])
        if url.scheme != "https" or url.netloc != "huggingface.co" or unquote(url.path) != expected_prefix + name:
            raise ValueError(f"Model source is not the pinned Hugging Face file: {name}")
        names.append(name)
    if len(files) != 22 or len(set(names)) != 22:
        raise ValueError("This deployment requires exactly 22 distinct manifest files")
    if not manifest.get("licenses"):
        raise ValueError("Model license files must be supplied")
    require_draft(repository, release_tag)
    staging = work / "staging"
    models = staging / "models/Qwen2.5-Omni-3B-MNN"
    models.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_with_retries, item, models / item["name"]) for item in files]
        for future in as_completed(futures):
            future.result()
    for item in files:
        target = models / item["name"]
        if target.stat().st_size != item["size_bytes"] or sha256(target) != item["sha256"]:
            raise ValueError(f"Final board-SHA verification failed: {item['name']}")
    for license_entry in manifest["licenses"]:
        source = REPO / member_path(license_entry["source"])
        target = staging / member_path(license_entry["path"])
        if source.is_symlink() or not source.is_file() or REPO not in source.resolve().parents:
            raise ValueError("License source must be a regular file within the repository")
        if target.exists():
            raise ValueError("License destination conflicts with another asset")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    metadata = staging / "metadata"
    metadata.mkdir()
    (metadata / "model-source.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    descriptor = package(staging, "model-mnn", work / "packages", release_tag, repository,
                         "当前部署 Qwen2.5-Omni-3B-MNN 的 22 个模型文件；逐文件匹配板端 SHA-256", True)
    result = json.loads(descriptor.read_text())
    upload = [descriptor.parent / part["name"] for part in result["parts"]] + [descriptor]
    require_draft(repository, release_tag)
    command = ["gh", "release", "upload", release_tag, *map(str, upload), "--repo", repository, "--clobber"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Release asset upload failed (gh exit {completed.returncode}); rerun the workflow")
    summary = {"package": result["id"], "model_files_verified": len(files), "parts_uploaded": len(result["parts"]),
               "compressed_bytes": sum(part["size_bytes"] for part in result["parts"]), "release_tag": release_tag}
    print(json.dumps(summary), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("## Model assets uploaded to draft Release\n\n")
            stream.write(f"Verified all {len(files)} model files against board SHA-256 values.\n\n")
            stream.write(f"Package: `{result['id']}`; parts: {len(result['parts'])}; tag: `{release_tag}`.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=REPO / "assets/model-source.json")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="model-release-") as temporary:
        publish(args.source_manifest, args.repository, args.release_tag, Path(temporary))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # urllib/subprocess exception repr can contain signed redirect URLs.
        if isinstance(error, (RuntimeError, ValueError)):
            raise SystemExit(f"Model publishing failed: {error}") from None
        raise SystemExit(f"Model publishing failed: {type(error).__name__}") from None
