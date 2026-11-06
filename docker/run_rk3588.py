#!/usr/bin/env python3
"""Run one explicit command with local assets and board devices; never choose devices automatically."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess


def make_command(args):
    assets, work, sdk = (p.expanduser().resolve() for p in (args.assets, args.work, args.rknn_sdk))
    mali = args.mali_lib.expanduser().resolve()
    for path in (assets, work, sdk):
        if not path.is_dir():
            raise ValueError(f"Directory does not exist: {path}")
    for path in (mali, sdk / "include/rknn_api.h", sdk / "lib/librknnrt.so"):
        if not path.is_file():
            raise ValueError(f"Missing dependency: {path}")
    if work == assets or work in assets.parents or assets in work.parents:
        raise ValueError("Use separate, non-nested asset and work directories")
    # Docker's --mount uses commas as delimiters; do not silently misparse paths.
    if any("," in str(p) for p in (assets, work, sdk, mali)):
        raise ValueError("Docker mount source paths must not contain commas")
    command = ["docker", "run", "--rm", "--platform", "linux/arm64", "--init",
               "--label", "kernel-bench.project=Qwen2.5-Omni-3B_VCB-Bench"]
    if args.interactive:
        command += ["-it"]
    for source, target, readonly in ((assets, "/assets", True), (work, "/work", False),
                                    (sdk, "/opt/board-rknn", True), (mali, "/opt/board-gpu/libmali.so.1", True)):
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    if not args.device:
        raise ValueError("Select the actual GPU/NPU device nodes with --device; no defaults are assumed")
    for value in args.device:
        path = Path(value)
        if not path.is_absolute() or not value.startswith("/dev/"):
            raise ValueError("--device must be an absolute /dev/ path")
        if not args.dry_run and not path.is_char_device():
            raise ValueError(f"Not an existing character device: {path}")
        command += ["--device", value]
    payload = args.command
    if payload and payload[0] == "--":
        payload = payload[1:]
    if not payload:
        raise ValueError("Supply an explicit command after -- (for example /bin/bash)")
    command += [args.image, *payload]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--assets", type=Path, required=True, help="Existing path on the Docker daemon host")
    p.add_argument("--work", type=Path, required=True, help="Existing writable directory, distinct from assets")
    p.add_argument("--rknn-sdk", type=Path, required=True, help="Directory containing include/ and lib/")
    p.add_argument("--mali-lib", type=Path, required=True, help="Board libmali providing OpenCL entry points")
    p.add_argument("--device", action="append", default=[], help="Explicit character device; repeat for GPU/NPU nodes")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    try:
        command = make_command(args)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
            raise ValueError("Run this launcher on the Linux/arm64 board with its local Docker daemon")
        # Record image identity before inference. No automatic git push or remote action.
        result = subprocess.run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
                                check=True, capture_output=True, text=True)
        record = {"image": args.image, "image_id": result.stdout.strip(), "argv": command,
                  "host": platform.platform(), "machine": platform.machine()}
        with (args.work.expanduser().resolve() / "container-invocations.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return subprocess.run(command).returncode
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        p.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
