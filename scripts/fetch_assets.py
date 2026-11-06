#!/usr/bin/env python3
"""Download, verify and unpack this experiment's GitHub Release assets.

Uses only the Python standard library. Interrupted downloads resume into .part
files; every complete part must match its manifest size and SHA-256. Archive
parts are read in manifest order without writing a second combined archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def simple_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"Invalid package or part name: {value!r}")
    return value


def verify_part(path: Path, part: dict) -> None:
    if path.stat().st_size != part["size_bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if sha256(path) != part["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path.name}; remove the corrupt file before retrying")


def download_part(part: dict, target: Path) -> Path:
    """Fetch one part, retaining interrupted bytes for the next invocation."""
    expected = part["size_bytes"]
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("Part size_bytes must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{64}", part["sha256"]):
        raise ValueError("Part sha256 must contain 64 lowercase hexadecimal digits")
    if urlsplit(part["url"]).scheme not in ("https", "http"):
        raise ValueError("Only HTTP(S) release asset URLs are supported")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"Download target cannot be a symlink: {target}")
    if target.exists():
        verify_part(target, part)
        print(f"verified cached: {target.name}", flush=True)
        return target
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"Partial download cannot be a symlink: {partial}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise ValueError(f"Partial download is larger than expected: {partial}")
    if offset < expected:
        headers = {"Accept-Encoding": "identity", "User-Agent": "kernel-bench-opt-assets/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"downloading: {target.name} ({offset}/{expected} bytes)", flush=True)
        request = Request(part["url"], headers=headers)
        with urlopen(request, timeout=120) as response:
            status = response.getcode()
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise ValueError("Missing or invalid Content-Range on resumed response")
                start, end, total = map(int, match.groups())
                if start != offset or total != expected or end != expected - 1:
                    raise ValueError("Content-Range does not match the requested asset suffix")
                mode = "ab" if offset else "wb"
            elif status == 200:
                # A server can ignore Range. Restart instead of appending a full response.
                offset, mode = 0, "wb"
            else:
                raise ValueError(f"Unexpected HTTP status: {status}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected - offset:
                raise ValueError("Content-Length does not match the expected remaining bytes")
            with partial.open(mode) as stream:
                count = offset
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    count += len(block)
                    if count > expected:
                        raise ValueError("Download exceeded its declared size")
                    stream.write(block)
    verify_part(partial, part)
    partial.replace(target)
    print(f"verified: {target.name}", flush=True)
    return target


class PartsReader:
    """Forward-only concatenation for tarfile's streaming reader."""

    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)
        self.stream = None

    def read(self, size: int) -> bytes:
        pieces = []
        remaining = size
        while remaining > 0:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open("rb")
            block = self.stream.read(remaining)
            if block:
                pieces.append(block)
                remaining -= len(block)
            else:
                self.stream.close()
                self.stream = None
        return b"".join(pieces)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def member_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    if not path.parts or path.parts[0] == ".downloads" or path.parts[0].startswith(".asset-staging-"):
        raise ValueError(f"Reserved or empty archive path: {name!r}")
    return Path(*path.parts)


def check_destination(output: Path, relative: Path) -> Path:
    destination = output / relative
    for path in [destination, *destination.parents]:
        if path == output:
            break
        if path.is_symlink():
            raise ValueError(f"Refusing an existing symlink destination: {path}")
        if path != destination and path.exists() and not path.is_dir():
            raise ValueError(f"Destination parent is not a directory: {path}")
    return destination


def unpack_parts(paths: list[Path], package: dict, output: Path) -> int:
    formats = {"tar": "r|", "tar.gz": "r|gz"}
    mode = formats.get(package["archive_format"])
    if mode is None:
        raise ValueError("archive_format must be tar or tar.gz")
    expected = package.get("extracted_size_bytes")
    if expected is not None and (not isinstance(expected, int) or expected < 0):
        raise ValueError("extracted_size_bytes must be a nonnegative integer")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    total = 0
    # Validate all members in isolation before writing into the final asset tree.
    with tempfile.TemporaryDirectory(prefix=".asset-staging-", dir=output) as temporary:
        staging = Path(temporary)
        reader = PartsReader(paths)
        try:
            with tarfile.open(fileobj=reader, mode=mode) as archive:
                for member in archive:
                    if member.isdir() and member.name in (".", "./"):
                        continue
                    relative = member_path(member.name)
                    if member.isdir():
                        (staging / relative).mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ValueError(f"Only regular files/directories are accepted: {member.name}")
                    if relative in seen:
                        raise ValueError(f"Duplicate archive file: {member.name}")
                    seen.add(relative)
                    total += member.size
                    if expected is not None and total > expected:
                        raise ValueError("Archive exceeds extracted_size_bytes")
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    with payload, target.open("xb") as stream:
                        shutil.copyfileobj(payload, stream, length=CHUNK)
                    if target.stat().st_size != member.size:
                        raise ValueError(f"Truncated archive member: {member.name}")
                    target.chmod(member.mode & 0o777)
                    records.append((relative, target))
        finally:
            reader.close()
        if expected is not None and total != expected:
            raise ValueError("Extracted bytes do not match extracted_size_bytes")
        for relative, target in records:
            destination = check_destination(output, relative)
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != target.stat().st_size or sha256(destination) != sha256(target):
                    raise ValueError(f"Existing asset differs; choose a new --output directory: {destination}")
        for relative, target in records:
            destination = check_destination(output, relative)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            target.replace(destination)
    return len(records)


def fetch_package(package: dict, output: Path) -> int:
    package_id = simple_name(package["id"])
    names = [simple_name(part["name"]) for part in package["parts"]]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate part names in {package_id}")
    downloads = output / ".downloads" / package_id
    check_destination(output, downloads.relative_to(output) / "placeholder")
    paths = [download_part(part, downloads / part["name"]) for part in package["parts"]]
    count = unpack_parts(paths, package, output)
    print(f"ready: {package_id} ({count} files)", flush=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "assets/manifest.json")
    parser.add_argument("--output", type=Path, default=REPO / "assets/local")
    parser.add_argument("--packages", nargs="+", help="Package IDs; defaults to packages marked default=true")
    parser.add_argument("--list", action="store_true", help="List available packages without downloading")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1:
        parser.error("Unsupported asset manifest schema_version")
    packages = manifest["packages"]
    by_id = {simple_name(package["id"]): package for package in packages}
    if len(by_id) != len(packages):
        parser.error("Duplicate package IDs")
    if args.list:
        for package in packages:
            size = sum(part["size_bytes"] for part in package["parts"])
            print(f"{package['id']}\t{size / 1024**2:.1f} MiB\tdefault={package.get('default', False)}\t{package.get('description', '')}")
        return 0
    selected = args.packages if args.packages is not None else [p["id"] for p in packages if p.get("default", False)]
    unknown = set(selected) - set(by_id)
    if unknown:
        parser.error("Unknown packages: " + ", ".join(sorted(unknown)))
    if not selected:
        parser.error("No packages selected; use --list and --packages")
    output = args.output.resolve()
    for package_id in dict.fromkeys(selected):
        fetch_package(by_id[package_id], output)
    print(f"Assets ready under {output}. No inference was executed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        raise SystemExit(f"Asset setup failed: {error}")
