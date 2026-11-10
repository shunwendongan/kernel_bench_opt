#!/usr/bin/env python3
"""Create deterministic, SHA-256 indexed GitHub Release asset parts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
from urllib.parse import quote

CHUNK = 1024 * 1024


class SplitWriter:
    def __init__(self, output: Path, package_id: str, maximum: int):
        self.output, self.package_id, self.maximum = output, package_id, maximum
        self.stream = None
        self.parts = []
        self.size = 0
        self.digest = None

    def write(self, data):
        view = memoryview(data)
        written = len(view)
        while view:
            if self.stream is None:
                name = f"{self.package_id}.tar.gz.part{len(self.parts):03d}"
                self.stream = (self.output / name).open("xb")
                self.name, self.size, self.digest = name, 0, hashlib.sha256()
            length = min(len(view), self.maximum - self.size)
            block = view[:length]
            self.stream.write(block)
            self.digest.update(block)
            self.size += length
            view = view[length:]
            if self.size == self.maximum:
                self.finish_part()
        return written

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def finish_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append({"name": self.name, "size_bytes": self.size, "sha256": self.digest.hexdigest()})
            self.stream = None


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size):
        block = self.stream.read(size)
        self.digest.update(block)
        self.count += len(block)
        return block


def package(source: Path, package_id: str, output: Path, release_tag: str,
            repository: str, description: str, default: bool = False,
            max_part_bytes: int = 1073741824) -> Path:
    from fetch_assets import simple_name
    simple_name(package_id)
    simple_name(release_tag)
    if len(repository.split("/")) != 2:
        raise ValueError("repository must be owner/repo")
    for item in repository.split("/"):
        simple_name(item)
    if max_part_bytes < 1:
        raise ValueError("max_part_bytes must be positive")
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or output == source or source in output.parents:
        raise ValueError("Source must be a directory and output must be outside it")
    output.mkdir(parents=True, exist_ok=True)
    descriptor = output / f"{package_id}.package.json"
    if descriptor.exists() or list(output.glob(f"{package_id}.tar.gz.part*")):
        raise ValueError("Package output already exists; choose an empty output directory")
    files = []
    for current, directories, names in os.walk(source, followlinks=False):
        for name in directories:
            if (Path(current) / name).is_symlink():
                raise ValueError("Source tree contains a symlink directory")
        for name in names:
            path = Path(current) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Only regular source files are supported")
            files.append(path)
    files.sort(key=lambda path: path.relative_to(source).as_posix())
    if not files:
        raise ValueError("Source contains no files")
    writer = SplitWriter(output, package_id, max_part_bytes)
    members = []
    try:
        with gzip.GzipFile(filename="", mode="wb", fileobj=writer, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for path in files:
                    before = path.stat()
                    name = path.relative_to(source).as_posix()
                    member = tarfile.TarInfo(name)
                    member.size = before.st_size
                    member.mode = 0o755 if before.st_mode & 0o111 else 0o644
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ""
                    with path.open("rb") as stream:
                        hashed = HashingReader(stream)
                        archive.addfile(member, hashed)
                    after = path.stat()
                    if hashed.count != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError(f"Source changed while packing: {name}")
                    members.append({"path": name, "size_bytes": hashed.count, "sha256": hashed.digest.hexdigest()})
    finally:
        writer.finish_part()
    base = f"https://github.com/{repository}/releases/download/{quote(release_tag, safe='')}"
    for part in writer.parts:
        part["url"] = base + "/" + quote(part["name"], safe="")
    result = {"id": package_id, "description": description, "archive_format": "tar.gz",
              "default": default, "parts": writer.parts,
              "extracted_size_bytes": sum(item["size_bytes"] for item in members),
              "file_count": len(members), "files": members,
              "packing": {"gzip_compresslevel": 1, "gzip_mtime": 0, "tar_format": "pax",
                          "order": "relative POSIX path", "metadata": "uid/gid/mtime=0; mode=0644 or 0755"}}
    descriptor.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"id": package_id, "file_count": len(members), "parts": len(writer.parts),
                      "compressed_bytes": sum(part["size_bytes"] for part in writer.parts),
                      "descriptor": str(descriptor)}), flush=True)
    return descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--default", action="store_true")
    parser.add_argument("--max-part-bytes", type=int, default=1073741824)
    args = parser.parse_args()
    package(args.source, args.id, args.output, args.release_tag, args.repository,
            args.description, args.default, args.max_part_bytes)


if __name__ == "__main__":
    main()
